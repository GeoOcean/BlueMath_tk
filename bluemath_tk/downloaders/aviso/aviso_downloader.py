"""
Downloader for AVISO+ altimetry products, served over FTP.

Products (configs in ``aviso/<PRODUCT>/<PRODUCT>_config.json``):

- SWOT: SWOT L2/L3 KaRIn expert products, one directory per cycle.
"""

import ftplib
import os
import posixpath
from typing import List, Optional

from .._base_downloaders import BaseDownloader, load_config
from .._download_result import DownloadResult

_HERE = os.path.dirname(__file__)


class AvisoDownloader(BaseDownloader):
    """
    Downloader for AVISO+ altimetry data, served over FTP (requires an account).

    Downloads all .nc files of the requested cycles of a dataset.
    The FTP connection is opened on first use, re-opened after failures, and
    closed with ``close()`` or when used as a context manager.

    Examples
    --------
    >>> with AvisoDownloader(
    ...     product="SWOT",
    ...     base_path_to_download="./swot_data",
    ...     username="your_username",
    ...     password="your_password"
    ... ) as downloader:
    ...     result = downloader.download_data(
    ...         dataset="swot-l3-expert",
    ...         cycles=["cycle_001"],
    ...         force=False,
    ...         dry_run=False
    ...     )
    """

    products_configs = {
        "SWOT": load_config(_HERE, "SWOT", "SWOT_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        username: str,
        password: str,
        debug: bool = True,
        **kwargs,
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
        **kwargs
            Passed to BaseDownloader (e.g., retries, retry_backoff).

        Raises
        ------
        ValueError
            If the product configuration is not found or FTP server is not specified.
        """

        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            debug=debug,
            **kwargs,
        )

        self.set_logger_name(
            f"AvisoDownloader-{product}", level="DEBUG" if debug else "INFO"
        )

        self._ftp_server = self.product_config.get("ftp_server")
        if self._ftp_server is None:
            raise ValueError("FTP server not found in product configuration")
        self._username = username
        self._password = password
        self._client: Optional[ftplib.FTP] = None
        self._exclude_attributes.extend(["_client", "_password"])

        self.logger.info(f"---- AVISO DOWNLOADER INITIALIZED ({product}) ----")

    def __enter__(self) -> "AvisoDownloader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def client(self) -> ftplib.FTP:
        """
        FTP client connection, opened and logged in on first use.

        Returns
        -------
        ftplib.FTP
            FTP client instance.
        """
        if self._client is None:
            self._client = ftplib.FTP(self._ftp_server, timeout=120)
            self._client.login(self._username, self._password)
        return self._client

    def close(self) -> None:
        """
        Close the FTP connection, if open.
        """
        if self._client is not None:
            try:
                self._client.quit()
            except Exception:
                self._client.close()
            self._client = None

    def download_data_swot(
        self,
        dataset: str,
        cycles: Optional[List[str]] = None,
        force: bool = False,
        dry_run: bool = True,
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
            If True, only check what would be downloaded. Default is True.

        Returns
        -------
        DownloadResult
            Result with all downloaded files and download statistics.

        Raises
        ------
        ValueError
            If dataset is not found or no cycles are available.
        """

        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )

        dataset_config = self.product_config["datasets"][dataset]
        ftp_base_path = dataset_config["ftp_base_path"]
        if cycles is None:
            cycles = dataset_config.get("cycles", [])
            if not cycles:
                raise ValueError(
                    f"No cycles specified for dataset '{dataset}' and cycles parameter not provided"
                )

        self.logger.info(f"Downloading dataset: {dataset}, cycles: {cycles}")
        result = self.create_download_result()

        for cycle in cycles:
            remote_dir = posixpath.join(ftp_base_path, cycle)
            try:
                files = self._list_nc_files(remote_dir)
            except Exception as e:
                self.logger.error(f"Error listing {remote_dir}: {e}")
                result.add_error(remote_dir, e)
                continue
            if not files:
                self.logger.warning(f"No files found in cycle {cycle}")
                continue

            for filename in files:
                local_path = os.path.join(
                    self.base_path_to_download, dataset, cycle, filename
                )
                remote_path = posixpath.join(remote_dir, filename)

                def fetch(part_file: str, remote_path: str = remote_path) -> None:
                    try:
                        with open(part_file, "wb") as f:
                            self.client.retrbinary(f"RETR {remote_path}", f.write)
                    except Exception:
                        # The connection may be broken: reconnect on next attempt
                        self.close()
                        raise

                self.fetch_file(local_path, fetch, result, force=force, dry_run=dry_run)

        return self.finalize_download_result(result)

    def _list_nc_files(self, remote_dir: str) -> List[str]:
        """
        List the .nc files (names only) in a directory of the FTP server.

        Parameters
        ----------
        remote_dir : str
            Absolute FTP directory (e.g., "/swot_products/.../Expert/cycle_001").

        Returns
        -------
        List[str]
            Sorted list of .nc filenames.
        """

        try:
            names = self.client.nlst(remote_dir)
        except ftplib.error_perm as e:
            # Some servers answer "550 No files found" for empty directories
            if str(e).startswith("550"):
                return []
            raise
        except (ftplib.error_temp, OSError, EOFError):
            self.close()
            names = self.client.nlst(remote_dir)

        return sorted(
            {posixpath.basename(name) for name in names if name.endswith(".nc")}
        )

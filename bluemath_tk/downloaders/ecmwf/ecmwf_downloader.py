"""
Downloader for ECMWF Open Data forecasts, through ``ecmwf-opendata``.

Products (configs in ``ecmwf/<PRODUCT>/<PRODUCT>_config.json``):

- OpenData: IFS and AIFS real-time forecasts.
"""

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict

import pandas as pd
from ecmwf.opendata import Client

from .._base_downloaders import BaseDownloader, load_config
from .._download_result import DownloadResult

_HERE = os.path.dirname(__file__)


class ECMWFDownloader(BaseDownloader):
    """
    Downloader for ECMWF Open Data real-time forecasts.

    Each call downloads one GRIB2 file for one forecast run, saved to
    base_path_to_download/product/dataset/model/resolution/<YYYYMMDDHH>/
    <params>_<steps>_<type>.grib2, so files of different runs never overwrite
    each other. Without ``date``, the latest available run is used.

    Examples
    --------
    >>> downloader = ECMWFDownloader(
    ...     product="OpenData",
    ...     base_path_to_download="./ecmwf_data",
    ...     model="ifs",
    ...     resolution="0p25",
    ...     source="aws",
    ... )
    >>> result = downloader.download_data(
    ...     dataset="forecast_data",
    ...     param=["msl"],
    ...     step=[0, 240],
    ...     type="fc",
    ...     force=False,
    ...     dry_run=False
    ... )
    """

    products_configs = {
        "OpenData": load_config(_HERE, "OpenData", "OpenData_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        model: str = "ifs",
        resolution: str = "0p25",
        source: str = "ecmwf",
        debug: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the ECMWFDownloader.

        Parameters
        ----------
        product : str
            The product to download data from. Currently only OpenData is supported.
        base_path_to_download : str
            Base path where downloaded files will be stored.
        model : str, optional
            The model to download data from (e.g., "ifs", "aifs"). Default is "ifs".
        resolution : str, optional
            The resolution to download data from (e.g., "0p25"). Default is "0p25".
        source : str, optional
            Where to download from: "ecmwf" (the ECMWF portal, limited to 500
            simultaneous connections) or one of its mirrors "aws", "azure",
            "google". Default is "ecmwf".
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        **kwargs
            Passed to BaseDownloader (e.g., retries, retry_backoff).

        Raises
        ------
        ValueError
            If the product configuration is not found, or if model/resolution are not supported.
        """

        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            debug=debug,
            **kwargs,
        )

        self.set_logger_name(
            f"ECMWFDownloader-{product}", level="DEBUG" if debug else "INFO"
        )

        # Validate model and resolution
        if model not in self.product_config["datasets"]["forecast_data"]["models"]:
            raise ValueError(f"Model {model} not supported for {self.product}")
        if (
            resolution
            not in self.product_config["datasets"]["forecast_data"]["resolutions"]
        ):
            raise ValueError(
                f"Resolution {resolution} not supported for {self.product}"
            )
        self._client = Client(
            source=source,
            model=model,
            resol=resolution,
            preserve_request_order=False,
            infer_stream_keyword=True,
        )
        self.logger.info(f"---- ECMWF DOWNLOADER INITIALIZED ({product}) ----")

        self.model = model
        self.resolution = resolution
        self.source = source

    @property
    def client(self) -> Client:
        """
        ECMWF OpenData client (initialized with model and resolution).

        Returns
        -------
        Client
            ECMWF OpenData client instance.
        """
        return self._client

    def download_data_open_data(
        self,
        dataset: str,
        force: bool = False,
        dry_run: bool = True,
        **kwargs,
    ) -> DownloadResult:
        """
        Download data for the OpenData product.

        Downloads one GRIB2 file with the requested parameters and steps of one
        forecast run. Files are saved to:
        base_path_to_download/product/dataset/model/resolution/<YYYYMMDDHH>/
        <params>_<steps>_<type>.grib2

        Parameters
        ----------
        dataset : str
            The dataset to download (e.g., "forecast_data").
            Use list_datasets() to see available datasets.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.
        **kwargs
            Keyword arguments passed to the ECMWF client retrieve method
            (e.g., param, step, type). ``date`` (YYYYMMDD, a datetime, or 0, -1,
            ... for days before today) and ``time`` (run hour) select the
            forecast run; without ``date`` the latest run is used. Finding the
            latest run needs a request to the server, also in dry runs.

        Returns
        -------
        DownloadResult
            Result with all downloaded files and download statistics.

        Raises
        ------
        ValueError
            If dataset is not found.
        """

        # Validate dataset
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )

        result = self.create_download_result()

        # Pin the forecast run, so the file always matches its path
        run = self._get_run_datetime(kwargs)
        kwargs["date"] = run.strftime("%Y%m%d")
        kwargs["time"] = run.hour

        variables = kwargs.get("param", [])
        if not isinstance(variables, list):
            variables = [variables]
        steps = kwargs.get("step", [])
        if not isinstance(steps, list):
            steps = [steps]
        type = kwargs.get("type", "fc")

        output_grib_file = os.path.join(
            self.base_path_to_download,
            self.product,
            dataset,
            self.model,
            self.resolution,
            run.strftime("%Y%m%d%H"),
            f"{'_'.join(variables)}_{'_'.join(str(step) for step in steps)}_{type}.grib2",
        )

        def fetch(part_file: str) -> None:
            self.client.retrieve(target=part_file, **kwargs)

        self.fetch_file(output_grib_file, fetch, result, force=force, dry_run=dry_run)

        return self.finalize_download_result(result)

    def _get_run_datetime(self, request: Dict[str, Any]) -> datetime:
        """
        Date and hour of the forecast run selected by a request.

        Parameters
        ----------
        request : Dict[str, Any]
            Retrieve keyword arguments. ``date`` may be missing (latest run), an
            integer (0 today, -1 yesterday, ...), "YYYYMMDD" or a date/datetime.
            ``time`` is the run hour (default 0).

        Returns
        -------
        datetime
            The run date and hour.
        """

        run_date = request.get("date")
        if run_date is None:
            latest_request = {
                key: value
                for key, value in request.items()
                if key not in ("date", "time")
            }
            return self.client.latest(**latest_request)

        if isinstance(run_date, int) and run_date <= 0:
            run_day = date.today() + timedelta(days=run_date)
        else:
            run_day = pd.Timestamp(str(run_date)).date()
        hour = int(str(request.get("time", 0)).split(":")[0])

        return datetime(run_day.year, run_day.month, run_day.day, hour)

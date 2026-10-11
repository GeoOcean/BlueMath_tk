"""
Downloader for the Copernicus Climate Data Store (CDS), through ``cdsapi``.

Products (configs in ``copernicus/<PRODUCT>/<PRODUCT>_config.json``):

- ERA5: global reanalysis (single levels, pressure levels, wave spectra).
- CERRA: regional reanalysis for Europe (single levels).

The CDS key is read from ~/.cdsapirc or the CDSAPI_KEY environment variable,
unless it is passed explicitly.
"""

import calendar
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cdsapi
import requests

from .._base_downloaders import BaseDownloader, load_config
from .._download_result import DownloadResult

_HERE = os.path.dirname(__file__)
ALL_MONTHS = [f"{month:02d}" for month in range(1, 13)]


class CopernicusDownloader(BaseDownloader):
    """
    Downloader for the Copernicus Climate Data Store (CDS).

    One file is written per variable and year when all 12 months are requested
    ({nc_name}_{YYYY}.nc), and one file per variable and month otherwise
    ({nc_name}_{YYYY}_{MM}.nc), so the same data never ends up in two files.
    Use file_period="month" to always write monthly files.

    Examples
    --------
    >>> downloader = CopernicusDownloader(
    ...     product="ERA5",
    ...     base_path_to_download="./copernicus_data",
    ...     api_key="your_api_key"
    ... )
    >>> result = downloader.download_data(
    ...     variables=["swh"],
    ...     years=["2020"],
    ...     months=["01"],
    ...     force=False,
    ...     dry_run=False
    ... )
    """

    products_configs = {
        "ERA5": load_config(_HERE, "ERA5", "ERA5_config.json"),
        "CERRA": load_config(_HERE, "CERRA", "CERRA_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        api_key: str = None,
        max_workers: int = 1,
        debug: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the CopernicusDownloader.

        Parameters
        ----------
        product : str
            The product to download data from (e.g., "ERA5", "CERRA").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        api_key : str, optional
            Copernicus CDS API key. If None, it is read by cdsapi from the
            CDSAPI_KEY environment variable or ~/.cdsapirc. Default is None.
        max_workers : int, optional
            Number of CDS requests submitted in parallel. CDS queues requests,
            so a few parallel requests reduce the waiting time a lot. Default is 1.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        **kwargs
            Passed to BaseDownloader (e.g., retries, retry_backoff).

        Raises
        ------
        ValueError
            If the product configuration is not found or server URL is not specified.
        """

        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            debug=debug,
            **kwargs,
        )

        self.set_logger_name(
            f"CopernicusDownloader-{product}", level="DEBUG" if debug else "INFO"
        )

        # Initialize CDS client
        server_url = self.product_config.get("url")
        if server_url is None:
            raise ValueError("Server URL not found in product configuration")
        self._server_url = server_url
        self._api_key = api_key
        self.max_workers = max_workers
        self._local = threading.local()
        self._exclude_attributes.extend(["_local", "_api_key"])
        # Create a client now so that missing credentials fail early
        self.client

        self.logger.info(f"---- COPERNICUS DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def client(self) -> cdsapi.Client:
        """
        CDS API client of the current thread (initialized with API key).

        Returns
        -------
        cdsapi.Client
            CDS API client instance.
        """
        if not hasattr(self._local, "client"):
            # Own session per thread (cdsapi shares one by default), and no
            # progress bars, which flood the logs of batch jobs
            self._local.client = cdsapi.Client(
                url=self._server_url,
                key=self._api_key,
                debug=self.debug,
                progress=False,
                session=requests.Session(),
            )
        return self._local.client

    def list_variables(self, type: str = None) -> List[str]:
        """
        List variables available for the product.

        Parameters
        ----------
        type : str, optional
            Filter by type (e.g., "ocean"). Default is None.

        Returns
        -------
        List[str]
            List of variable names.
        """

        if type is not None:
            return [
                var_name
                for var_name, var_info in self.product_config["variables"].items()
                if var_info["type"] == type
            ]

        return list(self.product_config["variables"].keys())

    def download_data_era5(
        self,
        variables: List[str],
        years: List[str],
        months: List[str],
        days: List[str] = None,
        times: List[str] = None,
        area: List[float] = None,
        product_type: str = "reanalysis",
        data_format: str = "netcdf",
        download_format: str = "unarchived",
        file_period: str = "auto",
        force: bool = False,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download ERA5 data.

        Downloads ERA5 reanalysis data for specified variables, time periods, and optionally
        a geographic area. Files are saved to:
        base_path_to_download/product/dataset/type/product_type/variable/filename.nc

        Parameters
        ----------
        variables : List[str]
            List of variable names to download. If empty, downloads all available variables.
        years : List[str]
            List of years to download (e.g., ["2020", "2021"]).
        months : List[str]
            List of months to download (e.g., ["01", "02"]).
        days : List[str], optional
            List of days to download. If None, downloads all days (1-31). Default is None.
        times : List[str], optional
            List of times to download (e.g., ["00:00", "12:00"]). If None, downloads all hours.
            Default is None.
        area : List[float], optional
            Geographic area as [north, west, south, east]. If None, downloads global data.
            Default is None.
        product_type : str, optional
            Product type (e.g., "reanalysis", "ensemble_mean"). Default is "reanalysis".
        data_format : str, optional
            Data format. Default is "netcdf".
        download_format : str, optional
            Download format. Default is "unarchived".
        file_period : str, optional
            "month" for one file per month, "year" for one file per year, or
            "auto" (yearly files if all 12 months are requested, monthly otherwise).
            Default is "auto".
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
            If years or months are empty lists.
        """

        variables, years, months = self._check_inputs(variables, years, months)
        if days is None:
            days = [f"{day:02d}" for day in range(1, 32)]
        if times is None:
            times = [f"{hour:02d}:00" for hour in range(24)]

        tasks = []
        for variable in variables:
            variable_config, dataset_config = self._get_variable_configs(variable)
            if variable_config is None:
                continue
            for year, period_months, period in self._split_periods(
                years, months, file_period
            ):
                request = copy.deepcopy(dataset_config["template"])
                if variable_config["dataset"] == "reanalysis-era5-complete":
                    last_day = calendar.monthrange(int(year), int(period_months[-1]))[1]
                    request["date"] = (
                        f"{year}-{period_months[0]}-01/to/"
                        f"{year}-{period_months[-1]}-{last_day:02d}"
                    )
                    if area is not None:
                        request["area"] = "/".join([str(coord) for coord in area])
                else:
                    request["variable"] = variable_config["cds_name"]
                    request["year"] = year
                    request["month"] = period_months
                    request["day"] = days
                    request["time"] = times
                    request["product_type"] = product_type
                    request["data_format"] = data_format
                    request["download_format"] = download_format
                    if area is not None:
                        request["area"] = area

                task = self._make_task(
                    variable,
                    variable_config,
                    dataset_config,
                    request,
                    year,
                    months,
                    period,
                    product_type,
                )
                if task is not None:
                    tasks.append(task)

        return self._run_tasks(tasks, force=force, dry_run=dry_run)

    def download_data_cerra(
        self,
        variables: List[str],
        years: List[str],
        months: List[str],
        days: List[str] = None,
        times: List[str] = None,
        area: List[float] = None,
        level_type: str = "surface_or_atmosphere",
        data_type: List[str] = None,
        product_type: str = "analysis",
        data_format: str = "netcdf",
        file_period: str = "auto",
        force: bool = False,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download CERRA data.

        Downloads CERRA reanalysis data for specified variables, time periods, and optionally
        a geographic area. Files are saved to:
        base_path_to_download/product/dataset/type/product_type/variable/filename.nc

        Parameters
        ----------
        variables : List[str]
            List of variable names to download. If empty, downloads all available variables.
        years : List[str]
            List of years to download (e.g., ["2020", "2021"]).
        months : List[str]
            List of months to download (e.g., ["01", "02"]).
        days : List[str], optional
            List of days to download. If None, downloads all days (1-31). Default is None.
        times : List[str], optional
            List of times to download (e.g., ["00:00", "12:00"]). If None, downloads standard
            times (00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00). Default is None.
        area : List[float], optional
            Geographic area as [north, west, south, east]. If None, downloads global data.
            Default is None.
        level_type : str, optional
            Level type (e.g., "surface_or_atmosphere"). Default is "surface_or_atmosphere".
        data_type : List[str], optional
            Data type (e.g., ["reanalysis"]). If None, uses ["reanalysis"]. Default is None.
        product_type : str, optional
            Product type (e.g., "analysis", "forecast"). Default is "analysis".
        data_format : str, optional
            Data format. Default is "netcdf".
        file_period : str, optional
            "month" for one file per month, "year" for one file per year, or
            "auto" (yearly files if all 12 months are requested, monthly otherwise).
            Default is "auto".
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
            If years or months are empty lists.
        """

        variables, years, months = self._check_inputs(variables, years, months)
        if days is None:
            days = [f"{day:02d}" for day in range(1, 32)]
        if times is None:
            times = [f"{hour:02d}:00" for hour in range(0, 24, 3)]
        if data_type is None:
            data_type = ["reanalysis"]

        tasks = []
        for variable in variables:
            variable_config, dataset_config = self._get_variable_configs(variable)
            if variable_config is None:
                continue
            for year, period_months, period in self._split_periods(
                years, months, file_period
            ):
                request = copy.deepcopy(dataset_config["template"])
                request["variable"] = [variable_config["cds_name"]]
                request["level_type"] = level_type
                request["data_type"] = data_type
                request["product_type"] = product_type
                request["year"] = [year]
                request["month"] = period_months
                request["day"] = days
                request["time"] = times
                request["data_format"] = data_format
                if area is not None:
                    request["area"] = area

                task = self._make_task(
                    variable,
                    variable_config,
                    dataset_config,
                    request,
                    year,
                    months,
                    period,
                    product_type,
                )
                if task is not None:
                    tasks.append(task)

        return self._run_tasks(tasks, force=force, dry_run=dry_run)

    def _check_inputs(
        self, variables: List[str], years: List[str], months: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate and format variables, years ("YYYY") and months ("MM").

        Any sequence is accepted (list, tuple, range, array).
        """

        variables = [] if variables is None else list(variables)
        if len(variables) == 0:
            variables = list(self.product_config["variables"].keys())
        years = [] if years is None or isinstance(years, str) else list(years)
        if len(years) == 0:
            raise ValueError("Years must be a non-empty list")
        months = [] if months is None or isinstance(months, str) else list(months)
        if len(months) == 0:
            raise ValueError("Months must be a non-empty list")
        years = [f"{int(year):04d}" for year in years]
        months = sorted({f"{int(month):02d}" for month in months})

        return variables, years, months

    @staticmethod
    def _split_periods(
        years: List[str], months: List[str], file_period: str = "auto"
    ) -> Iterator[Tuple[str, List[str], str]]:
        """
        Yield (year, months, period label) for each file to download.

        With file_period="auto", a full year gives one file per year,
        otherwise one file per month.
        """

        if file_period not in ("auto", "month", "year"):
            raise ValueError("file_period must be 'auto', 'month' or 'year'")

        for year in years:
            if file_period == "year" or (
                file_period == "auto" and months == ALL_MONTHS
            ):
                yield year, months, year
            else:
                for month in months:
                    yield year, [month], f"{year}_{month}"

    def _get_variable_configs(
        self, variable: str
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Get the variable and dataset configurations, or (None, None) if not found.
        """

        variable_config = self.product_config["variables"].get(variable)
        if variable_config is None:
            self.logger.error(f"Variable {variable} not found in configuration")
            return None, None

        dataset_config = self.product_config["datasets"].get(variable_config["dataset"])
        if dataset_config is None:
            self.logger.error(
                f"Dataset {variable_config['dataset']} not found in configuration"
            )
            return None, None

        return variable_config, dataset_config

    def _make_task(
        self,
        variable: str,
        variable_config: dict,
        dataset_config: dict,
        request: dict,
        year: str,
        months: List[str],
        period: str,
        product_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check the mandatory fields of a request and build its download task.

        Missing mandatory fields are taken from the variable configuration.

        Returns
        -------
        Optional[Dict[str, Any]]
            Task with dataset, request, output_file and legacy_file,
            or None if a mandatory field is missing.
        """

        for mandatory_field in dataset_config["mandatory_fields"]:
            if request.get(mandatory_field) is None:
                if mandatory_field not in variable_config:
                    self.logger.error(
                        f"Mandatory field {mandatory_field} not found for {variable}"
                    )
                    return None
                request[mandatory_field] = variable_config[mandatory_field]

        file_format = request.get("data_format", request.get("format", "netcdf"))
        extension = "grib" if file_format == "grib" else "nc"
        output_dir = os.path.join(
            self.base_path_to_download,
            self.product,
            variable_config["dataset"],
            variable_config["type"],
            product_type,
            variable_config["cds_name"],
        )
        nc_name = variable_config["nc_name"]

        # Files downloaded before the one-file-per-period naming
        legacy_file = None
        if months == ALL_MONTHS:
            legacy_file = os.path.join(
                output_dir, f"{nc_name}_{year}_{'_'.join(months)}.{extension}"
            )

        return {
            "variable": variable,
            "dataset": variable_config["dataset"],
            "request": request,
            "output_file": os.path.join(output_dir, f"{nc_name}_{period}.{extension}"),
            "legacy_file": legacy_file,
        }

    def _run_tasks(
        self, tasks: List[Dict[str, Any]], force: bool, dry_run: bool
    ) -> DownloadResult:
        """
        Download all tasks, with up to self.max_workers CDS requests in parallel.
        """

        result = self.create_download_result()
        if not tasks:
            return self.finalize_download_result(
                result, "No valid download tasks found"
            )

        self.logger.info(
            f"Prepared {len(tasks)} download tasks ({self.max_workers} in parallel)"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._run_task, task, force, dry_run) for task in tasks
            ]
            for future in futures:
                result.merge(future.result())

        return self.finalize_download_result(result)

    def _run_task(
        self, task: Dict[str, Any], force: bool, dry_run: bool
    ) -> DownloadResult:
        """
        Download a single task.
        """

        result = DownloadResult()
        legacy_file = task["legacy_file"]
        if not force and legacy_file and os.path.exists(legacy_file):
            result.add_skipped(legacy_file, "Already downloaded (legacy yearly file)")
            return result

        def fetch(part_file: str) -> None:
            self.client.retrieve(
                name=task["dataset"], request=task["request"], target=part_file
            )

        self.fetch_file(
            task["output_file"], fetch, result, force=force, dry_run=dry_run
        )

        return result

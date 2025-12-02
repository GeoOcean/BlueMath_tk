import calendar
import json
import os
from typing import Any, Dict, List, Optional

import cdsapi

from .._base_downloaders import BaseDownloader
from .._download_result import DownloadResult

config = {
    "url": "https://cds.climate.copernicus.eu/api",  # /v2?
    "key": "your-api-token",
}


class CopernicusDownloader(BaseDownloader):
    """
    This is the main class to download data from the Copernicus Climate Data Store.

    Attributes
    ----------
    product : str
        The product to download data from. Currently ERA5 and CERRA are supported.
    product_config : dict
        The configuration for the product to download data from.
    client : cdsapi.Client
        The client to interact with the Copernicus Climate Data Store API.

    Examples
    --------
    .. jupyter-execute::

        from bluemath_tk.downloaders.copernicus.copernicus_downloader import CopernicusDownloader

        # Example: Download ERA5 data
        copernicus_downloader = CopernicusDownloader(
            product="ERA5",
            base_path_to_download="/path/to/Copernicus/",  # Will be created if not available
            token=None,
        )
        result = copernicus_downloader.download_data_era5(
            variables=["swh"],
            years=["2020"],
            months=["01", "03"],
        )
        print(result)

        # Example: Download CERRA data
        cerra_downloader = CopernicusDownloader(
            product="CERRA",
            base_path_to_download="/path/to/Copernicus/",
            token=None,
        )
        result = cerra_downloader.download_data_cerra(
            variables=["10m_wind_speed"],
            years=["2020"],
            months=["01"],
            days=["01"],
        )
        print(result)

        # Or use dry_run to check what would be downloaded
        result = copernicus_downloader.download_data_era5(
            variables=["swh"],
            years=["2020"],
            months=["01", "03"],
            dry_run=True,  # Check without downloading
        )
        print(result)
    """

    products_configs = {
        "ERA5": json.load(
            open(os.path.join(os.path.dirname(__file__), "ERA5", "ERA5_config.json"))
        ),
        "CERRA": json.load(
            open(os.path.join(os.path.dirname(__file__), "CERRA", "CERRA_config.json"))
        ),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        token: str = None,
        debug: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """
        This is the constructor for the CopernicusDownloader class.

        Parameters
        ----------
        product : str
            The product to download data from. Currently ERA5 and CERRA are supported.
        base_path_to_download : str
            The base path to download the data to.
        token : str, optional
            The API token to use to download data. Default is None.
        debug : bool, optional
            Whether to run in debug mode. Default is True.
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
            f"CopernicusDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        # Always initialize client (will skip API calls in dry_run mode)
        self._client = cdsapi.Client(
            url=config["url"], key=token or config["key"], debug=self.debug
        )
        self.logger.info(f"---- COPERNICUS DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def product(self) -> str:
        return self._product

    @property
    def product_config(self) -> dict:
        return self._product_config

    @property
    def client(self) -> cdsapi.Client:
        return self._client

    def list_datasets(self) -> List[str]:
        """
        Lists the datasets available for the product.

        Returns
        -------
        List[str]
            The list of datasets available for the product.
        """

        return list(self.product_config["datasets"].keys())

    def list_variables(self, type: str = None) -> List[str]:
        """
        Lists the variables available for the product.
        Filtering by type if provided.

        Parameters
        ----------
        type : str, optional
            The type of variables to list. Default is None.

        Returns
        -------
        List[str]
            The list of variables available for the product.
        """

        if type == "ocean":
            return [
                var_name
                for var_name, var_info in self.product_config["variables"].items()
                if var_info["type"] == "ocean"
            ]

        return list(self.product_config["variables"].keys())

    def show_markdown_table(self) -> None:
        """
        Create a Markdown table from the configuration dictionary and print it.
        """

        # Define the table headers
        headers = ["name", "long_name", "units", "type"]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = (
            "| " + " | ".join(["-" * len(header) for header in headers]) + " |"
        )

        # Initialize the table with headers
        table_lines = [header_line, separator_line]

        # Add rows for each variable
        for var_name, var_info in self.product_config["variables"].items():
            long_name = var_info.get("long_name", "")
            units = var_info.get("units", "")
            type = var_info.get("type", "")
            row = f"| {var_name} | {long_name} | {units} | {type} |"
            table_lines.append(row)

        # Print the table
        print("\n".join(table_lines))

    def download_data(self, dry_run: bool = False, *args, **kwargs) -> DownloadResult:
        """
        Downloads the data for the product.

        Parameters
        ----------
        dry_run : bool, optional
            If True, only check what would be downloaded without actually downloading.
            Default is False.
        *args
            The arguments to pass to the download function.
        **kwargs
            The keyword arguments to pass to the download function.

        Returns
        -------
        DownloadResult
            The download result with information about downloaded, skipped, and error files.

        Raises
        ------
        ValueError
            If the product is not supported.
        """

        if self.product == "ERA5":
            return self.download_data_era5(dry_run=dry_run, *args, **kwargs)
        elif self.product == "CERRA":
            return self.download_data_cerra(dry_run=dry_run, *args, **kwargs)
        else:
            raise ValueError(f"Download for product {self.product} not supported")

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
        force: bool = False,
        num_workers: int = 1,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Downloads the data for the ERA5 product.

        Parameters
        ----------
        variables : List[str]
            The variables to download. If not provided, all variables in self.product_config
            will be downloaded.
        years : List[str]
            The years to download. Years are downloaded one by one.
        months : List[str]
            The months to download. Months are downloaded together.
        days : List[str], optional
            The days to download. If None, all days in the month will be downloaded.
            Default is None.
        times : List[str], optional
            The times to download. If None, all times in the day will be downloaded.
            Default is None.
        area : List[float], optional
            The area to download. If None, the whole globe will be downloaded.
            Default is None.
        product_type : str, optional
            The product type to download. Default is "reanalysis".
        data_format : str, optional
            The data format to download. Default is "netcdf".
        download_format : str, optional
            The download format to use. Default is "unarchived".
        force : bool, optional
            Whether to force the download. Default is False.
        num_workers : int, optional
            Number of parallel workers for downloading. Default is 1 (sequential).
            Set to > 1 to enable parallel downloads. Note: CDS API has rate limits.

        Returns
        -------
        DownloadResult
            The download result with information about downloaded, skipped, and error files.

        Notes
        -----
        - Parallel downloads are I/O-bound, so ThreadPoolExecutor is used.
        - CDS API has rate limits (typically 20 concurrent requests), so be careful
          with num_workers > 20.
        """

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Input validation
        if not isinstance(variables, list):
            raise ValueError("Variables must be a list of strings")
        elif len(variables) == 0:
            variables = list(self.product_config["variables"].keys())
            self.logger.info(f"Variables not provided. Using {variables}")
        if not isinstance(years, list) or len(years) == 0:
            raise ValueError("Years must be a non-empty list of strings")
        else:
            years = [f"{int(year):04d}" for year in years]
        if not isinstance(months, list) or len(months) == 0:
            raise ValueError("Months must be a non-empty list of strings")
        else:
            months = [f"{int(month):02d}" for month in months]
            last_month = months[-1]
        if days is not None:
            if not isinstance(days, list) or len(days) == 0:
                raise ValueError("Day must be a non-empty list of strings")
        else:
            days = [f"{day:02d}" for day in range(1, 32)]
            self.logger.info(f"Day not provided. Using {days}")
        if times is not None:
            if not isinstance(times, list) or len(times) == 0:
                raise ValueError("Time must be a non-empty list of strings")
        else:
            times = [f"{hour:02d}:00" for hour in range(24)]
            self.logger.info(f"Time not provided. Using {times}")
        if area is not None:
            if not isinstance(area, list) or len(area) != 4:
                raise ValueError("Area must be a list of 4 floats")
        if not isinstance(product_type, str):
            raise ValueError("Product type must be a string")
        if not isinstance(data_format, str):
            raise ValueError("Data format must be a string")
        if not isinstance(download_format, str):
            raise ValueError("Download format must be a string")
        if not isinstance(force, bool):
            raise ValueError("Force must be a boolean")
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")

        # Initialize download result
        result = self.create_download_result()

        # Prepare download tasks
        download_tasks = []
        for variable in variables:
            for year in years:
                task = self._prepare_era5_download_task(
                    variable=variable,
                    year=year,
                    months=months,
                    days=days,
                    times=times,
                    area=area,
                    product_type=product_type,
                    data_format=data_format,
                    download_format=download_format,
                    last_month=last_month,
                )
                if task is not None:
                    download_tasks.append(task)

        if not download_tasks:
            self.logger.warning("No valid download tasks prepared")
            return self.finalize_download_result(
                result, "No valid download tasks found"
            )

        if dry_run:
            self.logger.info(f"DRY RUN: Checking {len(download_tasks)} files for ERA5")

        self.logger.info(
            f"Prepared {len(download_tasks)} download tasks. "
            f"Using {num_workers} worker(s) for parallel execution."
        )

        # Execute downloads (parallel or sequential)
        if num_workers > 1 and not dry_run:
            # Parallel execution
            results_dict = self.parallel_execute(
                func=self._download_single_file,
                items=download_tasks,
                num_workers=min(num_workers, len(download_tasks)),
                cpu_intensive=False,  # I/O bound, use threads
                force=force,
                dry_run=dry_run,
            )
            # Aggregate results
            for task_result in results_dict.values():
                if isinstance(task_result, DownloadResult):
                    result.downloaded_files.extend(task_result.downloaded_files)
                    result.skipped_files.extend(task_result.skipped_files)
                    result.error_files.extend(task_result.error_files)
                    result.errors.extend(task_result.errors)
        else:
            # Sequential execution with progress bar
            iterator = download_tasks
            if self.show_progress and tqdm is not None and not dry_run:
                iterator = tqdm(
                    download_tasks,
                    desc="Downloading ERA5 data",
                    unit="file",
                )

            for task in iterator:
                task_result = self._download_single_file(
                    task, force=force, dry_run=dry_run
                )
                if isinstance(task_result, DownloadResult):
                    result.downloaded_files.extend(task_result.downloaded_files)
                    result.skipped_files.extend(task_result.skipped_files)
                    result.error_files.extend(task_result.error_files)
                    result.errors.extend(task_result.errors)

        # Finalize and return result
        return self.finalize_download_result(result)

    def _prepare_era5_download_task(
        self,
        variable: str,
        year: str,
        months: List[str],
        days: List[str],
        times: List[str],
        area: Optional[List[float]],
        product_type: str,
        data_format: str,
        download_format: str,
        last_month: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a download task dictionary for a single ERA5 variable-year combination.

        Returns None if the task cannot be prepared (e.g., missing config).
        """

        variable_config = self.product_config["variables"].get(variable)
        if variable_config is None:
            self.logger.error(
                f"Variable {variable} not found in product configuration file"
            )
            return None

        variable_dataset = self.product_config["datasets"].get(
            variable_config["dataset"]
        )
        if variable_dataset is None:
            self.logger.error(
                f"Dataset {variable_config['dataset']} not found in product configuration file"
            )
            return None

        template_for_variable = variable_dataset["template"].copy()
        if variable == "spectra":
            template_for_variable["date"] = (
                f"{year}-{months[0]}-01/to/{year}-{months[-1]}-31"
            )
            if area is not None:
                template_for_variable["area"] = "/".join([str(coord) for coord in area])
        else:
            template_for_variable["variable"] = variable_config["cds_name"]
            template_for_variable["year"] = year
            template_for_variable["month"] = months
            template_for_variable["day"] = days
            template_for_variable["time"] = times
            template_for_variable["product_type"] = product_type
            template_for_variable["data_format"] = data_format
            template_for_variable["download_format"] = download_format
            if area is not None:
                template_for_variable["area"] = area

        # Check mandatory fields
        for mandatory_field in variable_dataset["mandatory_fields"]:
            try:
                if template_for_variable.get(mandatory_field) is None:
                    template_for_variable[mandatory_field] = variable_config[
                        mandatory_field
                    ]
            except KeyError:
                self.logger.error(
                    f"Mandatory field {mandatory_field} not found in variable configuration file for {variable}"
                )
                return None

        # Create output file path
        output_nc_file = os.path.join(
            self.base_path_to_download,
            self.product,
            variable_config["dataset"],
            variable_config["type"],
            product_type,
            variable_config["cds_name"],
            f"{variable_config['nc_name']}_{year}_{'_'.join(months)}.nc",
        )

        return {
            "variable": variable,
            "year": year,
            "variable_config": variable_config,
            "variable_dataset": variable_dataset,
            "template": template_for_variable,
            "output_file": output_nc_file,
            "last_month": last_month,
        }

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
        force: bool = False,
        num_workers: int = 1,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Downloads the data for the CERRA product.

        Parameters
        ----------
        variables : List[str]
            The variables to download. If not provided, all variables in self.product_config
            will be downloaded.
        years : List[str]
            The years to download. Years are downloaded one by one.
        months : List[str]
            The months to download. Months are downloaded together.
        days : List[str], optional
            The days to download. If None, all days in the month will be downloaded.
            Default is None.
        times : List[str], optional
            The times to download. If None, default CERRA times (3-hourly) will be used.
            Default is None.
        area : List[float], optional
            The area to download. If None, the whole domain will be downloaded.
            Default is None.
        level_type : str, optional
            The level type. Default is "surface_or_atmosphere".
        data_type : List[str], optional
            The data type. Default is ["reanalysis"].
        product_type : str, optional
            The product type to download. Default is "analysis".
        data_format : str, optional
            The data format to download. Default is "netcdf".
        force : bool, optional
            Whether to force the download. Default is False.
        num_workers : int, optional
            Number of parallel workers for downloading. Default is 1 (sequential).
            Set to > 1 to enable parallel downloads. Note: CDS API has rate limits.
        dry_run : bool, optional
            If True, only check what would be downloaded without actually downloading.
            Default is False.

        Returns
        -------
        DownloadResult
            The download result with information about downloaded, skipped, and error files.

        Notes
        -----
        - Parallel downloads are I/O-bound, so ThreadPoolExecutor is used.
        - CDS API has rate limits (typically 20 concurrent requests), so be careful
          with num_workers > 20.
        - CERRA data is available from September 1984 to present.
        - Default times are 3-hourly (00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00).
        """

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Input validation
        if not isinstance(variables, list):
            raise ValueError("Variables must be a list of strings")
        elif len(variables) == 0:
            variables = list(self.product_config["variables"].keys())
            self.logger.info(f"Variables not provided. Using {variables}")
        if not isinstance(years, list) or len(years) == 0:
            raise ValueError("Years must be a non-empty list of strings")
        else:
            years = [f"{int(year):04d}" for year in years]
        if not isinstance(months, list) or len(months) == 0:
            raise ValueError("Months must be a non-empty list of strings")
        else:
            months = [f"{int(month):02d}" for month in months]
            last_month = months[-1]
        if days is not None:
            if not isinstance(days, list) or len(days) == 0:
                raise ValueError("Days must be a non-empty list of strings")
            days = [f"{int(day):02d}" for day in days]
        else:
            days = [f"{day:02d}" for day in range(1, 32)]
            self.logger.info("Days not provided. Using all days in month")
        if times is not None:
            if not isinstance(times, list) or len(times) == 0:
                raise ValueError("Times must be a non-empty list of strings")
        else:
            # Default CERRA times: 3-hourly
            times = [
                "00:00",
                "03:00",
                "06:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ]
            self.logger.info(f"Times not provided. Using default CERRA times: {times}")
        if area is not None:
            if not isinstance(area, list) or len(area) != 4:
                raise ValueError("Area must be a list of 4 floats")
        if data_type is None:
            data_type = ["reanalysis"]
        if not isinstance(data_type, list):
            raise ValueError("Data type must be a list of strings")
        if not isinstance(level_type, str):
            raise ValueError("Level type must be a string")
        if not isinstance(product_type, str):
            raise ValueError("Product type must be a string")
        if not isinstance(data_format, str):
            raise ValueError("Data format must be a string")
        if not isinstance(force, bool):
            raise ValueError("Force must be a boolean")
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")

        # Initialize download result
        result = self.create_download_result()

        # Prepare download tasks
        download_tasks = []
        for variable in variables:
            for year in years:
                task = self._prepare_cerra_download_task(
                    variable=variable,
                    year=year,
                    months=months,
                    days=days,
                    times=times,
                    area=area,
                    level_type=level_type,
                    data_type=data_type,
                    product_type=product_type,
                    data_format=data_format,
                    last_month=last_month,
                )
                if task is not None:
                    download_tasks.append(task)

        if not download_tasks:
            self.logger.warning("No valid download tasks prepared")
            return self.finalize_download_result(
                result, "No valid download tasks found"
            )

        if dry_run:
            self.logger.info(f"DRY RUN: Checking {len(download_tasks)} files for CERRA")

        self.logger.info(
            f"Prepared {len(download_tasks)} download tasks. "
            f"Using {num_workers} worker(s) for parallel execution."
        )

        # Execute downloads (parallel or sequential)
        if num_workers > 1 and not dry_run:
            # Parallel execution
            results_dict = self.parallel_execute(
                func=self._download_single_file,
                items=download_tasks,
                num_workers=min(num_workers, len(download_tasks)),
                cpu_intensive=False,  # I/O bound, use threads
                force=force,
                dry_run=dry_run,
            )
            # Aggregate results
            for task_result in results_dict.values():
                if isinstance(task_result, DownloadResult):
                    result.downloaded_files.extend(task_result.downloaded_files)
                    result.skipped_files.extend(task_result.skipped_files)
                    result.error_files.extend(task_result.error_files)
                    result.errors.extend(task_result.errors)
        else:
            # Sequential execution with progress bar
            iterator = download_tasks
            if self.show_progress and tqdm is not None and not dry_run:
                iterator = tqdm(
                    download_tasks,
                    desc="Downloading CERRA data",
                    unit="file",
                )

            for task in iterator:
                task_result = self._download_single_file(
                    task, force=force, dry_run=dry_run
                )
                if isinstance(task_result, DownloadResult):
                    result.downloaded_files.extend(task_result.downloaded_files)
                    result.skipped_files.extend(task_result.skipped_files)
                    result.error_files.extend(task_result.error_files)
                    result.errors.extend(task_result.errors)

        # Finalize and return result
        return self.finalize_download_result(result)

    def _prepare_cerra_download_task(
        self,
        variable: str,
        year: str,
        months: List[str],
        days: List[str],
        times: List[str],
        area: Optional[List[float]],
        level_type: str,
        data_type: List[str],
        product_type: str,
        data_format: str,
        last_month: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a download task for CERRA data.

        Parameters
        ----------
        variable : str
            Variable name.
        year : str
            Year to download.
        months : List[str]
            Months to download.
        days : List[str]
            Days to download.
        times : List[str]
            Times to download.
        area : Optional[List[float]]
            Area to download.
        level_type : str
            Level type.
        data_type : List[str]
            Data type.
        product_type : str
            Product type.
        data_format : str
            Data format.
        last_month : str
            Last month in the list.

        Returns
        -------
        Optional[Dict[str, Any]]
            Download task dictionary or None if invalid.
        """

        # Get variable configuration
        variable_config = self.product_config["variables"].get(variable)
        if variable_config is None:
            self.logger.error(f"Variable {variable} not found in configuration")
            return None

        # Get dataset configuration
        variable_dataset = self.product_config["datasets"].get(
            variable_config["dataset"]
        )
        if variable_dataset is None:
            self.logger.error(
                f"Dataset {variable_config['dataset']} not found in configuration"
            )
            return None

        # Create template for CERRA request
        template_for_variable = variable_dataset["template"].copy()
        template_for_variable["variable"] = [variable_config["cds_name"]]
        template_for_variable["level_type"] = level_type
        template_for_variable["data_type"] = data_type
        template_for_variable["product_type"] = product_type
        template_for_variable["year"] = [year]
        template_for_variable["month"] = months
        template_for_variable["day"] = days
        template_for_variable["time"] = times
        template_for_variable["data_format"] = data_format

        if area is not None:
            template_for_variable["area"] = area

        # Check mandatory fields
        for mandatory_field in variable_dataset["mandatory_fields"]:
            if template_for_variable.get(mandatory_field) is None:
                self.logger.error(
                    f"Mandatory field {mandatory_field} not found in template for {variable}"
                )
                return None

        # Create output file path
        output_nc_file = os.path.join(
            self.base_path_to_download,
            self.product,
            variable_config["dataset"],
            variable_config["type"],
            product_type,
            variable_config["cds_name"],
            f"{variable_config['nc_name']}_{year}_{'_'.join(months)}.nc",
        )

        return {
            "variable": variable,
            "year": year,
            "variable_config": variable_config,
            "template": template_for_variable,
            "last_month": last_month,
            "output_file": output_nc_file,
        }

    def _download_single_file(
        self, task: Dict[str, Any], force: bool = False, dry_run: bool = False
    ) -> DownloadResult:
        """
        Download a single file based on a task dictionary.

        This method handles file checking, downloading with retry, and error handling.
        """

        result = DownloadResult()
        output_file = task["output_file"]
        variable = task["variable"]
        variable_config = task["variable_config"]
        template = task["template"]
        last_month = task["last_month"]
        year = task["year"]

        # Create output directory if needed
        if not dry_run:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Check if file exists and is complete
            if not force and (dry_run or os.path.exists(output_file)):
                if os.path.exists(output_file):
                    # Check file completeness
                    _, last_day = calendar.monthrange(int(year), int(last_month))
                    expected_end_time = f"{year}-{last_month}-{last_day}T23"
                    is_complete, reason = self.check_file_complete(
                        output_file,
                        expected_time_range=(None, expected_end_time),
                    )

                    if is_complete:
                        if dry_run:
                            result.add_skipped(
                                output_file, "File already complete (dry run)"
                            )
                        else:
                            result.add_downloaded(output_file)
                        return result
                    else:
                        # File exists but is incomplete
                        self.logger.debug(
                            f"{output_file} exists but is incomplete: {reason}"
                        )
                        if dry_run:
                            result.add_skipped(
                                output_file, f"Incomplete: {reason} (dry run)"
                            )
                            return result
                        # Will re-download below
                elif dry_run:
                    result.add_skipped(output_file, "File does not exist (dry run)")
                    return result

            # Download the file (with retry mechanism)
            if dry_run:
                result.add_skipped(output_file, f"Would download {variable} (dry run)")
                return result

            self.logger.debug(f"Downloading: {variable} to {output_file}")

            def _retrieve():
                self.client.retrieve(
                    name=variable_config["dataset"],
                    request=template,
                    target=output_file,
                )

            self.retry_with_backoff(
                _retrieve,
                error_message=f"Failed to download {output_file}",
            )
            result.add_downloaded(output_file)

        except Exception as e:
            self.logger.error(f"Error downloading {output_file}: {e}")
            result.add_error(output_file, e)

        return result

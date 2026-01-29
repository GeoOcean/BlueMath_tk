import json
import os
from typing import Any, Dict, List, Optional

import cdsapi

from .._base_downloaders import BaseDownloader
from .._download_result import DownloadResult


class CopernicusDownloader(BaseDownloader):
    """
    Simple downloader for Copernicus Climate Data Store.

    Examples
    --------
    >>> downloader = CopernicusDownloader(
    ...     product="ERA5",
    ...     base_path_to_download="./copernicus_data",
    ...     token="your_token"
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
        api_key: str,
        debug: bool = True,
    ) -> None:
        """
        Initialize the CopernicusDownloader.

        Parameters
        ----------
        product : str
            The product to download data from (e.g., "ERA5", "CERRA").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        api_key : str
            Copernicus CDS API key.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found or server URL is not specified.
        """

        super().__init__(
            product=product, base_path_to_download=base_path_to_download, debug=debug
        )

        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            raise ValueError(
                f"Product '{product}' not found. Available: {list(self.products_configs.keys())}"
            )

        self.set_logger_name(
            f"CopernicusDownloader-{product}", level="DEBUG" if debug else "INFO"
        )

        # Initialize CDS client
        server_url = self._product_config.get("url")
        if server_url is None:
            raise ValueError("Server URL not found in product configuration")
        self._client = cdsapi.Client(url=server_url, key=api_key, debug=self.debug)

        self.logger.info(f"---- COPERNICUS DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def product_config(self) -> dict:
        """
        Product configuration dictionary loaded from config file.

        Returns
        -------
        dict
            Product configuration dictionary.
        """
        return self._product_config

    @property
    def client(self) -> cdsapi.Client:
        """
        CDS API client (initialized with API key).

        Returns
        -------
        cdsapi.Client
            CDS API client instance.
        """
        return self._client

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

        if type == "ocean":
            return [
                var_name
                for var_name, var_info in self.product_config["variables"].items()
                if var_info["type"] == "ocean"
            ]

        return list(self.product_config["variables"].keys())

    def download_data(
        self,
        dry_run: bool = True,
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
            Default is True.
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

        if not isinstance(variables, list) or len(variables) == 0:
            variables = list(self.product_config["variables"].keys())
        if not isinstance(years, list) or len(years) == 0:
            raise ValueError("Years must be a non-empty list")
        years = [f"{int(year):04d}" for year in years]
        if not isinstance(months, list) or len(months) == 0:
            raise ValueError("Months must be a non-empty list")
        months = [f"{int(month):02d}" for month in months]
        last_month = months[-1]
        if days is None:
            days = [f"{day:02d}" for day in range(1, 32)]
        if times is None:
            times = [f"{hour:02d}:00" for hour in range(24)]

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
            return self.finalize_download_result(
                result, "No valid download tasks found"
            )

        self.logger.info(f"Prepared {len(download_tasks)} download tasks")

        # Download files sequentially
        for task in download_tasks:
            task_result = self._download_single_file(task, force=force, dry_run=dry_run)
            if isinstance(task_result, DownloadResult):
                result.downloaded_files.extend(task_result.downloaded_files)
                result.skipped_files.extend(task_result.skipped_files)
                result.error_files.extend(task_result.error_files)
                result.errors.extend(task_result.errors)

        return self.finalize_download_result(result)

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

        if not isinstance(variables, list) or len(variables) == 0:
            variables = list(self.product_config["variables"].keys())
        if not isinstance(years, list) or len(years) == 0:
            raise ValueError("Years must be a non-empty list")
        years = [f"{int(year):04d}" for year in years]
        if not isinstance(months, list) or len(months) == 0:
            raise ValueError("Months must be a non-empty list")
        months = [f"{int(month):02d}" for month in months]
        last_month = months[-1]
        if days is None:
            days = [f"{day:02d}" for day in range(1, 32)]
        if times is None:
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
        if data_type is None:
            data_type = ["reanalysis"]

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
            return self.finalize_download_result(
                result, "No valid download tasks found"
            )

        self.logger.info(f"Prepared {len(download_tasks)} download tasks")

        # Download files sequentially
        for task in download_tasks:
            task_result = self._download_single_file(task, force=force, dry_run=dry_run)
            if isinstance(task_result, DownloadResult):
                result.downloaded_files.extend(task_result.downloaded_files)
                result.skipped_files.extend(task_result.skipped_files)
                result.error_files.extend(task_result.error_files)
                result.errors.extend(task_result.errors)

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
        Prepare a download task for ERA5.

        Creates a task dictionary with all necessary information for downloading
        a single variable for a single year.

        Parameters
        ----------
        variable : str
            Variable name.
        year : str
            Year (formatted as "YYYY").
        months : List[str]
            List of months (formatted as "MM").
        days : List[str]
            List of days (formatted as "DD").
        times : List[str]
            List of times (formatted as "HH:MM").
        area : Optional[List[float]]
            Geographic area as [north, west, south, east] or None.
        product_type : str
            Product type.
        data_format : str
            Data format.
        download_format : str
            Download format.
        last_month : str
            Last month in the list (used for date range formatting).

        Returns
        -------
        Optional[Dict[str, Any]]
            Task dictionary with download information, or None if configuration is invalid.
        """

        variable_config = self.product_config["variables"].get(variable)
        if variable_config is None:
            self.logger.error(f"Variable {variable} not found in configuration")
            return None

        variable_dataset = self.product_config["datasets"].get(
            variable_config["dataset"]
        )
        if variable_dataset is None:
            self.logger.error(
                f"Dataset {variable_config['dataset']} not found in configuration"
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
            if template_for_variable.get(mandatory_field) is None:
                try:
                    template_for_variable[mandatory_field] = variable_config[
                        mandatory_field
                    ]
                except KeyError:
                    self.logger.error(
                        f"Mandatory field {mandatory_field} not found for {variable}"
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
        Prepare a download task for CERRA.

        Creates a task dictionary with all necessary information for downloading
        a single variable for a single year.

        Parameters
        ----------
        variable : str
            Variable name.
        year : str
            Year (formatted as "YYYY").
        months : List[str]
            List of months (formatted as "MM").
        days : List[str]
            List of days (formatted as "DD").
        times : List[str]
            List of times (formatted as "HH:MM").
        area : Optional[List[float]]
            Geographic area as [north, west, south, east] or None.
        level_type : str
            Level type.
        data_type : List[str]
            Data type list.
        product_type : str
            Product type.
        data_format : str
            Data format.
        last_month : str
            Last month in the list (used for date range formatting).

        Returns
        -------
        Optional[Dict[str, Any]]
            Task dictionary with download information, or None if configuration is invalid.
        """

        variable_config = self.product_config["variables"].get(variable)
        if variable_config is None:
            self.logger.error(f"Variable {variable} not found in configuration")
            return None

        variable_dataset = self.product_config["datasets"].get(
            variable_config["dataset"]
        )
        if variable_dataset is None:
            self.logger.error(
                f"Dataset {variable_config['dataset']} not found in configuration"
            )
            return None

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
                    f"Mandatory field {mandatory_field} not found for {variable}"
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
        self, task: Dict[str, Any], force: bool = False, dry_run: bool = True
    ) -> DownloadResult:
        """
        Download a single file based on a task dictionary.

        Parameters
        ----------
        task : Dict[str, Any]
            Task dictionary containing download information (output_file, template, etc.).
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.

        Returns
        -------
        DownloadResult
            Result with information about the downloaded, skipped, or error file.
        """

        result = DownloadResult()
        output_file = task["output_file"]
        variable = task["variable"]
        variable_config = task["variable_config"]
        template = task["template"]

        if not dry_run:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Check if file already exists
            if not force and os.path.exists(output_file):
                if dry_run:
                    result.add_skipped(output_file, "File already exists (dry run)")
                else:
                    result.add_downloaded(output_file)
                return result

            if dry_run:
                result.add_skipped(output_file, f"Would download {variable} (dry run)")
                return result

            # Download file
            self.logger.debug(f"Downloading: {variable} to {output_file}")
            self.client.retrieve(
                name=variable_config["dataset"],
                request=template,
                target=output_file,
            )
            result.add_downloaded(output_file)
            self.logger.info(f"Downloaded: {output_file}")

        except Exception as e:
            self.logger.error(f"Error downloading {output_file}: {e}")
            result.add_error(output_file, e)

        return result

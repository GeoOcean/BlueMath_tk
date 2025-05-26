import calendar
import json
import os
from typing import List

import cdsapi
import xarray as xr

from .._base_downloaders import BaseDownloader

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
        The product to download data from. Currently only ERA5 is supported.
    product_config : dict
        The configuration for the product to download data from.
    client : cdsapi.Client
        The client to interact with the Copernicus Climate Data Store API.

    Examples
    --------
    .. jupyter-execute::

        from bluemath_tk.downloaders.copernicus.copernicus_downloader import CopernicusDownloader

        copernicus_downloader = CopernicusDownloader(
            product="ERA5",
            base_path_to_download="/path/to/Copernicus/",  # Will be created if not available
            token=None,
            check=True,
        )
        result = copernicus_downloader.download_data_era5(
            variables=["swh"],
            years=["2020"],
            months=["01", "03"],
        )
        print(result)
    """

    products_configs = {
        "ERA5": json.load(
            open(os.path.join(os.path.dirname(__file__), "ERA5", "ERA5_config.json"))
        )
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        token: str = None,
        debug: bool = True,
        check: bool = True,
    ) -> None:
        """
        This is the constructor for the CopernicusDownloader class.

        Parameters
        ----------
        product : str
            The product to download data from. Currently only ERA5 is supported.
        base_path_to_download : str
            The base path to download the data to.
        token : str, optional
            The API token to use to download data. Default is None.
        debug : bool, optional
            Whether to run in debug mode. Default is True.
        check : bool, optional
            Whether to just check the data. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found.
        """

        super().__init__(
            base_path_to_download=base_path_to_download, debug=debug, check=check
        )
        self._product = product
        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            raise ValueError(f"{product} configuration not found")
        self.set_logger_name(
            f"CopernicusDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        if not self.check:
            self._client = cdsapi.Client(
                url=config["url"], key=token or config["key"], debug=self.debug
            )
            self.logger.info("---- DOWNLOADING DATA ----")
        else:
            self.logger.info("---- CHECKING DATA ----")

    @property
    def product(self) -> str:
        return self._product

    @property
    def product_config(self) -> dict:
        return self._product_config

    @property
    def client(self) -> cdsapi.Client:
        return self._client

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

    def list_datasets(self) -> List[str]:
        """
        Lists the datasets available for the product.

        Returns
        -------
        List[str]
            The list of datasets available for the product.
        """

        return list(self.product_config["datasets"].keys())

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

    def download_data(self, *args, **kwargs) -> str:
        """
        Downloads the data for the product.

        Parameters
        ----------
        *args
            The arguments to pass to the download function.
        **kwargs
            The keyword arguments to pass to the download function.

        Returns
        -------
        str
            The message with the fully downloaded files and the not fully downloaded files.

        Raises
        ------
        ValueError
            If the product is not supported.
        """

        if self.product == "ERA5":
            return self.download_data_era5(*args, **kwargs)
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
    ) -> str:
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

        Returns
        -------
        str
            The message with the fully downloaded files and the not fully downloaded files.
            Error files are also included.

        TODO
        -----
        - Implement lambda function to name the files.
        """

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

        fully_downloaded_files: List[str] = []
        NOT_fullly_downloaded_files: List[str] = []
        error_files: List[str] = []

        for variable in variables:
            for year in years:
                variable_config = self.product_config["variables"].get(variable)
                if variable_config is None:
                    self.logger.error(
                        f"Variable {variable} not found in product configuration file"
                    )
                    continue
                variable_dataset = self.product_config["datasets"].get(
                    variable_config["dataset"]
                )
                if variable_dataset is None:
                    self.logger.error(
                        f"Dataset {variable_config['dataset']} not found in product configuration file"
                    )
                    continue

                template_for_variable = variable_dataset["template"].copy()
                if variable == "spectra":
                    template_for_variable["date"] = (
                        f"{year}-{months[0]}-01/to/{year}-{months[-1]}-31"
                    )
                    if area is not None:
                        template_for_variable["area"] = "/".join(
                            [str(coord) for coord in area]
                        )
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

                self.logger.info(
                    f"""
                    Template for variable {variable}:
                    {template_for_variable}
                    """
                )

                skip_because_of_manadatory_fields = False
                for mandatory_field in variable_dataset["mandatory_fields"]:
                    try:
                        if template_for_variable.get(mandatory_field) is None:
                            template_for_variable[mandatory_field] = variable_config[
                                mandatory_field
                            ]
                    except KeyError:
                        self.logger.error(
                            f"Mandotory field {mandatory_field} not found in variable configuration file for {variable}"
                        )
                        skip_because_of_manadatory_fields = True
                if skip_because_of_manadatory_fields:
                    continue

                # Create the output file name once request is properly formatted
                output_nc_file = os.path.join(
                    self.base_path_to_download,
                    self.product,
                    variable_config["dataset"],
                    variable_config["type"],
                    product_type,
                    variable_config["cds_name"],
                    f"{variable_config['nc_name']}_{year}_{'_'.join(months)}.nc",
                    # f"era5_waves_{variable_config['cds_name']}_{year}.nc",
                )
                # Create the output directory if it does not exist
                if not self.check:
                    os.makedirs(os.path.dirname(output_nc_file), exist_ok=True)

                self.logger.info(f"""
                                 
                    Analyzing {output_nc_file}

                """)

                try:
                    if self.check or not force:
                        if os.path.exists(output_nc_file):
                            self.logger.debug(
                                f"Checking {output_nc_file} file is complete"
                            )
                            try:
                                nc = xr.open_dataset(output_nc_file)
                                _, last_day = calendar.monthrange(
                                    int(year), int(last_month)
                                )
                                last_hour = f"{year}-{last_month}-{last_day}T23"
                                try:
                                    last_hour_nc = str(nc.time[-1].values)
                                except Exception as _te:
                                    last_hour_nc = str(nc.valid_time[-1].values)
                                nc.close()
                                if last_hour not in last_hour_nc:
                                    self.logger.debug(
                                        f"{output_nc_file} ends at {last_hour_nc} instead of {last_hour}"
                                    )
                                    if self.check:
                                        NOT_fullly_downloaded_files.append(
                                            output_nc_file
                                        )
                                    else:
                                        self.logger.debug(
                                            f"Downloading: {variable} to {output_nc_file} because it is not complete"
                                        )
                                        self.client.retrieve(
                                            name=variable_config["dataset"],
                                            request=template_for_variable,
                                            target=output_nc_file,
                                        )
                                        fully_downloaded_files.append(output_nc_file)
                                else:
                                    self.logger.debug(
                                        f"{output_nc_file} already downloaded and complete"
                                    )
                                    fully_downloaded_files.append(output_nc_file)
                            except Exception as e:
                                self.logger.error(
                                    f"Error was raised opening {output_nc_file} - {e}, re-downloading..."
                                )
                                if self.check:
                                    NOT_fullly_downloaded_files.append(output_nc_file)
                                else:
                                    self.logger.debug(
                                        f"Downloading: {variable} to {output_nc_file} because it is not complete"
                                    )
                                    self.client.retrieve(
                                        name=variable_config["dataset"],
                                        request=template_for_variable,
                                        target=output_nc_file,
                                    )
                                    fully_downloaded_files.append(output_nc_file)
                        elif self.check:
                            NOT_fullly_downloaded_files.append(output_nc_file)
                        else:
                            self.logger.debug(
                                f"Downloading: {variable} to {output_nc_file}"
                            )
                            self.client.retrieve(
                                name=variable_config["dataset"],
                                request=template_for_variable,
                                target=output_nc_file,
                            )
                            fully_downloaded_files.append(output_nc_file)
                    else:
                        self.logger.debug(
                            f"Downloading: {variable} to {output_nc_file}"
                        )
                        self.client.retrieve(
                            name=variable_config["dataset"],
                            request=template_for_variable,
                            target=output_nc_file,
                        )
                        fully_downloaded_files.append(output_nc_file)

                except Exception as e:
                    self.logger.error(f"""
                                        
                        Skippping {output_nc_file} for {e}

                    """)
                    error_files.append(output_nc_file)

        fully_downloaded_files_str = "\n".join(fully_downloaded_files)
        NOT_fullly_downloaded_files_str = "\n".join(NOT_fullly_downloaded_files)
        error_files = "\n".join(error_files)

        return f"""
            Fully downloaded files:
            {fully_downloaded_files_str}
            Not fully downloaded files:
            {NOT_fullly_downloaded_files_str}
            Error files:
            {error_files}
        """

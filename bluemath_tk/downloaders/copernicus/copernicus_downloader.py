import os
import json
from typing import List
import calendar
import cdsapi
import xarray as xr
from ..base_downloaders import BlueMathDownloader

config = {
    "url": "https://cds.climate.copernicus.eu/api",  # /v2?
    "key": "5cf7efae-13fc-4085-8a98-80d82bdb55f5",
}


class CopernicusDownloader(BlueMathDownloader):
    products_configs = {
        "ERA5": json.load(
            open(os.path.join(os.path.dirname(__file__), "ERA5", "ERA5_config.json"))
        )
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        debug: bool = True,
    ) -> None:
        super().__init__(base_path_to_download=base_path_to_download, debug=debug)
        self._product = product
        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            raise ValueError(f"{product} configuration not found")
        self.set_logger_name(f"CopernicusDownloader - {product}")
        # self._client = cdsapi.Client(
        #     url=config["url"], key=config["key"], debug=self.debug
        # )

    @property
    def product(self) -> str:
        return self._product

    @property
    def product_config(self) -> dict:
        return self._product_config

    @property
    def client(self) -> cdsapi.Client:
        return self._client

    def download_data(self, *args, **kwargs):
        if self.product == "ERA5":
            return self.download_data_era5(*args, **kwargs)
        else:
            raise ValueError(f"Download for product {self.product} not supported")

    def check_data(self, *args, **kwargs):
        return super().check_data(*args, **kwargs)

    def list_variables(self):
        return list(self.product_config["variables"].keys())

    def list_datasets(self):
        return list(self.product_config["datasets"].keys())

    def download_data_era5(
        self,
        variables: List[str],
        years: List[str],
        months: List[str],
        day: List[str] = None,
        time: List[str] = None,
        product_type: str = "reanalysis",
        data_format: str = "netcdf",
        download_format: str = "unarchived",
        force: bool = False,
    ):
        if not isinstance(variables, list) or len(variables) == 0:
            raise ValueError("Variables must be a non-empty list of strings")
        if not isinstance(years, list) or len(years) == 0:
            raise ValueError("Years must be a non-empty list of strings")
        if not isinstance(months, list) or len(months) == 0:
            raise ValueError("Months must be a non-empty list of strings")
        if day is not None:
            if not isinstance(day, list) or len(day) == 0:
                raise ValueError("Day must be a non-empty list of strings")
        else:
            day = [f"{day:02d}" for day in range(1, 32)]
            self.logger.info(f"Day not provided. Using {day}")
        if time is not None:
            if not isinstance(time, list) or len(time) == 0:
                raise ValueError("Time must be a non-empty list of strings")
        else:
            time = [f"{hour:02d}:00" for hour in range(24)]
            self.logger.info(f"Time not provided. Using {time}")
        if not isinstance(product_type, str):
            raise ValueError("Product type must be a string")
        if not isinstance(data_format, str):
            raise ValueError("Data format must be a string")
        if not isinstance(download_format, str):
            raise ValueError("Download format must be a string")
        if not isinstance(force, bool):
            raise ValueError("Force must be a boolean")

        for variable in variables:
            for year in years:
                for month in months:
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
                    template_for_variable["variable"] = variable_config["cds_name"]
                    template_for_variable["year"] = year
                    template_for_variable["month"] = month
                    template_for_variable["day"] = day
                    template_for_variable["time"] = time
                    template_for_variable["product_type"] = product_type
                    template_for_variable["data_format"] = data_format
                    template_for_variable["download_format"] = download_format

                    for mandatory_field in variable_dataset["mandatory_fields"]:
                        try:
                            if template_for_variable.get(mandatory_field) is None:
                                template_for_variable[mandatory_field] = (
                                    variable_config[mandatory_field]
                                )
                        except KeyError as e:
                            raise KeyError(
                                f"Mandotory field {mandatory_field} not found in variable configuration file for {variable}"
                            ) from e

                    # Create the output file name once request is properly formatted
                    output_nc_file = os.path.join(
                        self.base_path_to_download,
                        self.product,
                        variable_config["dataset"],
                        variable_config["type"],
                        product_type,
                        variable_config["nc_name"],
                        year,
                        f"{variable_config["nc_name"]}_{year}{month}.nc",
                    )
                    # Create the output directory if it does not exist
                    os.makedirs(os.path.dirname(output_nc_file), exist_ok=True)

                    continue

                    self.logger.info(
                        f"Downloading variable {variable}: {json.dumps(template_for_variable, indent=2)}"
                    )
                    self.logger.info(
                        f"Variable info: {json.dumps(variable_config, indent=2)}"
                    )

                    if os.path.exists(output_nc_file):
                        self.logger.debug(f"File {output_nc_file} exists")
                        if not force:
                            self.logger.debug("Checking the file is complete")
                            nc = xr.open_dataset(output_nc_file)
                            _, last_day = calendar.monthrange(int(year), int(month))
                            last_hour = f"{year}-{int(month):02d}-{last_day}T23"
                            last_hour_nc = str(nc.time[-1].values)
                            nc.close()
                            if last_hour not in last_hour_nc:
                                self.logger.debug(
                                    f"{output_nc_file} ends at {last_hour_nc} instead of {last_hour}"
                                )
                                self.logger.debug(
                                    f"Downloading: {variable} to {output_nc_file}"
                                )
                                self.client.retrieve(
                                    name=variable_config["dataset"],
                                    request=template_for_variable,
                                    target=output_nc_file,
                                )
                        else:
                            self.logger.debug(
                                f"Downloading: {variable} to {output_nc_file}"
                            )
                            self.client.retrieve(
                                name=variable_config["dataset"],
                                request=template_for_variable,
                                target=output_nc_file,
                            )
                    else:
                        self.logger.debug(
                            f"Downloading: {variable} to {output_nc_file}"
                        )
                        self.client.retrieve(
                            name=variable_config["dataset"],
                            request=template_for_variable,
                            target=output_nc_file,
                        )


if __name__ == "__main__":
    copernicus_downloader = CopernicusDownloader(
        product="ERA5",
        base_path_to_download="/home/tausiaj/DATA/Copernicus/",
        debug=True,
    )
    copernicus_downloader.download_data(
        variables=["geo500", "tp", "p140122"], years=["2021"], months=["01"]
    )

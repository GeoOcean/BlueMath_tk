import os
import calendar
import cdsapi
import xarray as xr
import logging
from .base_downloader import BlueMathDownloader


class CopernicusDownloader(BlueMathDownloader):
    def __init__(self, base_path, variables):
        super().__init__(base_pat=base_path)
        self.set_logger_name("CopernicusDownloader")
        self._client = cdsapi.Client()
        self._variables = variables

    @property
    def client(self):
        return self._client
    
    @client.setter
    def client(self, value):
        self._client = value

    @property
    def variables(self):
        return self._variables
    
    @variables.setter
    def variables(self, value):
        self._variables = value

    def download_data(self, variable, year, month, check=False, force=False, dbg=False):
        if dbg:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug(f"Downloading {variable} for {year}-{month}")
        if len(variable) == 0:
            variable = list(self.variables.keys())

        plantilla = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": "variable",
            "year": "year",
            "month": "month",
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
        }

        for var in variable:
            var_dict = self.variables[var]
            plantilla["variable"] = var_dict["cds_name"]
            plantilla["year"] = year
            plantilla["month"] = month
            source_retrieval = var_dict["source"]

            nc_file = f"{self.base_path}{var_dict['path']}{var}/{var_dict['short_name']}_{year}{int(month):02d}.nc"
            self.logger.debug(f"Checking: {nc_file}")
            if not force:
                if os.path.exists(nc_file):
                    self.logger.debug(f"File {nc_file} exists")
                    nc = xr.open_dataset(nc_file)
                    _, last_day = calendar.monthrange(int(year), int(month))
                    last_hour = f"{year}-{int(month):02d}-{last_day}T23"
                    last_hour_nc = str(nc.time[-1].values)
                    nc.close()
                    if last_hour not in last_hour_nc:
                        self.logger.info(
                            f"{nc_file} ends at {last_hour_nc} instead of {last_hour}"
                        )
                        if not check:
                            self.logger.debug(f" Downloading: {var} to {nc_file}")
                            self.client.retrieve(source_retrieval, plantilla, nc_file)
                else:
                    self.logger.debug(f" Downloading: {var} to {nc_file}")
                    self.client.retrieve(source_retrieval, plantilla, nc_file)
            else:
                self.logger.debug(f" Downloading: {var} to {nc_file}")
                self.client.retrieve(source_retrieval, plantilla, nc_file)

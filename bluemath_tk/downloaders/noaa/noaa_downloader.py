import gzip
import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import requests
import xarray as xr

from .._base_downloaders import BaseDownloader


class NOAADownloader(BaseDownloader):
    """
    This is the main class to download and read data from NOAA.

    Attributes
    ----------
    config : dict
        The configuration for NOAA data sources loaded from JSON file.
    base_path_to_download : Path
        Base path where the data is stored.
    debug : bool
        Whether to run in debug mode.

    Examples
    --------
    .. jupyter-execute::

        from bluemath_tk.downloaders.noaa.noaa_downloader import NOAADownloader

        noaa_downloader = NOAADownloader(
            base_path_to_download="/path/to/NOAA/",  # Will be created if not available
            debug=True,
            check=False,
        )

        # Download buoy bulk parameters and load DataFrame
        result = noaa_downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2020, 2021, 2022],
            load_df=True
        )
        print(result)
    """

    config = json.load(
        open(os.path.join(os.path.dirname(__file__), "NOAA_config.json"))
    )

    def __init__(
        self,
        base_path_to_download: str,
        debug: bool = True,
        check: bool = False,
    ) -> None:
        """
        Initialize the NOAA downloader.

        Parameters
        ----------
        base_path_to_download : str
            The base path to download the data to.
        debug : bool, optional
            Whether to run in debug mode. Default is True.
        check : bool, optional
            Whether to just check the data. Default is False.
        """

        super().__init__(
            base_path_to_download=base_path_to_download, debug=debug, check=check
        )
        self.set_logger_name("NOAADownloader", level="DEBUG" if debug else "INFO")

        if not self.check:
            self.logger.info("---- DOWNLOADING NOAA DATA ----")
        else:
            self.logger.info("---- CHECKING NOAA DATA ----")

    @property
    def datasets(self) -> dict:
        return self.config["datasets"]

    @property
    def data_types(self) -> dict:
        return self.config["data_types"]

    def list_data_types(self) -> List[str]:
        """
        Lists the available data types.

        Returns
        -------
        List[str]
            The list of available data types.
        """

        return list(self.data_types.keys())

    def list_datasets(self) -> List[str]:
        """
        Lists the available datasets.

        Returns
        -------
        List[str]
            The list of available datasets.
        """

        return list(self.datasets.keys())

    def show_markdown_table(self) -> None:
        """
        Create a Markdown table from the configuration dictionary and print it.
        """

        # Define the table headers
        headers = ["name", "long_name", "description", "dataset"]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = (
            "| " + " | ".join(["-" * len(header) for header in headers]) + " |"
        )

        # Initialize the table with headers
        table_lines = [header_line, separator_line]

        # Add rows for each data type
        for data_type_name, data_type_info in self.data_types.items():
            name = data_type_info.get("name", "")
            long_name = data_type_info.get("long_name", "")
            description = data_type_info.get("description", "")
            dataset = data_type_info.get("dataset", "")
            row = f"| {name} | {long_name} | {description} | {dataset} |"
            table_lines.append(row)

        # Print the table
        print("\n".join(table_lines))

    def download_data(
        self, data_type: str, load_df: bool = False, **kwargs
    ) -> Union[pd.DataFrame, xr.Dataset, str]:
        """
        Downloads the data for the specified data type.

        Parameters
        ----------
        data_type : str
            The data type to download.
            - 'bulk_parameters'
            - 'wave_spectra'
            - 'directional_spectra'
            - 'wind_forecast'

        load_df : bool, optional
            Whether to load and return the DataFrame after downloading.
            Default is False.
            If True and multiple years are specified, all years will be combined
            into a single DataFrame.
        **kwargs
            Additional keyword arguments specific to each data type.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, str]
            Downloaded data or status message.

        Raises
        ------
        ValueError
            If the data type is not supported.
        """

        if data_type not in self.data_types:
            raise ValueError(
                f"Data type {data_type} not supported. Available types: {self.list_data_types()}"
            )

        data_type_config = self.data_types[data_type]
        dataset_config = self.datasets[data_type_config["dataset"]]

        result = None
        if data_type == "bulk_parameters":
            result = self._download_bulk_parameters(
                data_type_config, dataset_config, **kwargs
            )
            if load_df:
                buoy_id = kwargs.get("buoy_id")
                years = kwargs.get("years", [])
                if years:
                    result = self.read_bulk_parameters(buoy_id, years)
        elif data_type == "wave_spectra":
            result = self._download_wave_spectra(
                data_type_config, dataset_config, **kwargs
            )
            if load_df:
                buoy_id = kwargs.get("buoy_id")
                years = kwargs.get("years", [])
                if years:
                    result = self.read_wave_spectra(buoy_id, years)
        elif data_type == "directional_spectra":
            result = self._download_directional_spectra(
                data_type_config, dataset_config, **kwargs
            )
            if load_df:
                buoy_id = kwargs.get("buoy_id")
                years = kwargs.get("years", [])
                if years:
                    result = self.read_directional_spectra(buoy_id, years)
        elif data_type == "wind_forecast":
            result = self._download_wind_forecast(
                data_type_config, dataset_config, **kwargs
            )
        else:
            raise ValueError(f"Download for data type {data_type} not implemented")

        return result

    def _download_bulk_parameters(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download bulk parameters for a specific buoy and years.

        Parameters
        ----------
        data_type_config : dict
            The configuration for the data type.
        dataset_config : dict
            The configuration for the dataset.
        buoy_id : str
            The buoy ID.
        years : List[int]
            The years to download data for.

        Returns
        -------
        pd.DataFrame
            The downloaded data.
        """

        self.logger.info(
            f"Downloading bulk parameters for buoy {buoy_id}, years {years}"
        )

        all_data = []
        base_url = dataset_config["base_url"]

        for year in years:
            # Try main URL first, then fallbacks
            urls = [
                f"{base_url}/{data_type_config['url_pattern'].format(buoy_id=buoy_id, year=year)}"
            ]
            for fallback in data_type_config.get("fallback_urls", []):
                urls.append(f"{base_url}/{fallback.format(buoy_id=buoy_id, year=year)}")

            df = self._download_single_year_bulk(
                urls, data_type_config["columns"], year
            )
            if df is not None:
                all_data.append(df)
                self.logger.info(f"Buoy {buoy_id}: Data found for year {year}")
            else:
                self.logger.warning(
                    f"Buoy {buoy_id}: No data available for year {year}"
                )

        if all_data:
            # Combine all years
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(["YYYY", "MM", "DD", "hh"])

            # Save to CSV if not in check mode
            if not self.check:
                buoy_dir = os.path.join(
                    self.base_path_to_download, "buoy_data", buoy_id
                )
                os.makedirs(buoy_dir, exist_ok=True)
                output_file = os.path.join(
                    buoy_dir, f"buoy_{buoy_id}_bulk_parameters.csv"
                )
                combined_df.to_csv(output_file, index=False)
                self.logger.info(f"Data saved to {output_file}")

                return f"Data saved to {output_file}"

            return combined_df
        else:
            self.logger.error(f"No data found for buoy {buoy_id}")
            return None

    def _download_single_year_bulk(
        self, urls: List[str], columns: List[str], year: int
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse bulk parameters for a single year.

        Parameters
        ----------
        urls : List[str]
            The URLs to download the data from.
        columns : List[str]
            The columns to read from the data.
        year : int
            The year to download data for.

        Returns
        -------
        Optional[pd.DataFrame]
            The downloaded data.
        """

        for url in urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    content = gzip.decompress(response.content).decode("utf-8")

                    # Skip the header rows and read the data
                    data = []
                    lines = content.split("\n")[2:]  # Skip first two lines (headers)

                    # Check format by looking at the first data line
                    first_line = next(line for line in lines if line.strip())
                    cols = first_line.split()

                    # Determine format based on number of columns and year format
                    has_minutes = len(cols) == 18  # Post-2012 format has 18 columns

                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if parts:
                                # Convert 2-digit year to 4 digits if needed
                                if int(parts[0]) < 100:
                                    parts[0] = str(int(parts[0]) + 1900)

                                # Add minutes column if it doesn't exist
                                if not has_minutes:
                                    parts.insert(4, "00")

                                data.append(" ".join(parts))

                    # Read the modified data
                    df = pd.read_csv(
                        io.StringIO("\n".join(data)),
                        sep=r"\s+",
                        names=columns,
                    )

                    # Validate dates
                    valid_dates = (
                        (df["MM"] >= 1)
                        & (df["MM"] <= 12)
                        & (df["DD"] >= 1)
                        & (df["DD"] <= 31)
                        & (df["hh"] >= 0)
                        & (df["hh"] <= 23)
                        & (df["mm"] >= 0)
                        & (df["mm"] <= 59)
                    )

                    df = df[valid_dates].copy()

                    if len(df) > 0:
                        return df

            except Exception as e:
                self.logger.debug(f"Failed to download from {url}: {e}")
                continue

        return None

    def read_bulk_parameters(
        self, buoy_id: str, years: Union[int, List[int]]
    ) -> Optional[pd.DataFrame]:
        """
        Read bulk parameters for a specific buoy and year(s).

        Parameters
        ----------
        buoy_id : str
            The buoy ID.
        years : Union[int, List[int]]
            The year(s) to read data for. Can be a single year or a list of years.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the bulk parameters, or None if data not found.
        """

        if isinstance(years, int):
            years = [years]

        all_data = []
        for year in years:
            file_path = os.path.join(
                self.base_path_to_download,
                "buoy_data",
                buoy_id,
                f"buoy_{buoy_id}_bulk_parameters.csv",
            )
            try:
                df = pd.read_csv(file_path)
                df["datetime"] = pd.to_datetime(
                    df["YYYY"].astype(str)
                    + "-"
                    + df["MM"].astype(str).str.zfill(2)
                    + "-"
                    + df["DD"].astype(str).str.zfill(2)
                    + " "
                    + df["hh"].astype(str).str.zfill(2)
                    + ":"
                    + df["mm"].astype(str).str.zfill(2)
                )
                all_data.append(df)
            except FileNotFoundError:
                self.logger.error(
                    f"No bulk parameters file found for buoy {buoy_id} year {year}"
                )

        if all_data:
            return pd.concat(all_data, ignore_index=True).sort_values("datetime")
        return None

    def _download_wave_spectra(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        **kwargs,
    ) -> str:
        """
        Download wave spectra data for a specific buoy.

        Parameters
        ----------
        data_type_config : dict
            The configuration for the data type.
        dataset_config : dict
            The configuration for the dataset.
        buoy_id : str
            The buoy ID.
        years : List[int]
            The years to download data for.

        Returns
        -------
        str
            The status message.
        """

        self.logger.info(f"Downloading wave spectra for buoy {buoy_id}, years {years}")

        base_url = dataset_config["base_url"]
        buoy_dir = os.path.join(
            self.base_path_to_download, "buoy_data", buoy_id, "wave_spectra"
        )

        if not self.check:
            os.makedirs(buoy_dir, exist_ok=True)

        downloaded_files = []

        for year in years:
            url = f"{base_url}/{data_type_config['url_pattern'].format(buoy_id=buoy_id, year=year)}"

            try:
                # Read the data
                df = pd.read_csv(
                    url,
                    compression="gzip",
                    sep=r"\s+",
                    na_values=["MM", "99.00", "999.0"],
                )

                # Skip if empty or invalid data
                if df.empty or len(df.columns) < 5:
                    self.logger.warning(f"No valid data for {buoy_id} - {year}")
                    continue

                # Process datetime (simplified version)
                if not self.check:
                    output_file = os.path.join(
                        buoy_dir, f"buoy_{buoy_id}_spectra_{year}.csv"
                    )
                    df.to_csv(output_file, index=False)
                    downloaded_files.append(output_file)
                    self.logger.info(f"Successfully saved data for {buoy_id} - {year}")

            except Exception as e:
                self.logger.warning(f"No data found for: {buoy_id} - {year}: {e}")
                continue

        return f"Downloaded {len(downloaded_files)} files for wave spectra"

    def read_wave_spectra(
        self, buoy_id: str, years: Union[int, List[int]]
    ) -> Optional[pd.DataFrame]:
        """
        Read wave spectra data for a specific buoy and year(s).

        Parameters
        ----------
        buoy_id : str
            The buoy ID.
        years : Union[int, List[int]]
            The year(s) to read data for. Can be a single year or a list of years.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the wave spectra, or None if data not found
        """

        if isinstance(years, int):
            years = [years]

        all_data = []
        for year in years:
            file_path = os.path.join(
                self.base_path_to_download,
                "buoy_data",
                buoy_id,
                "wave_spectra",
                f"buoy_{buoy_id}_spectra_{year}.csv",
            )
            try:
                df = pd.read_csv(file_path)
                try:
                    df["date"] = pd.to_datetime(
                        df[["YYYY", "MM", "DD", "hh"]].rename(
                            columns={
                                "YYYY": "year",
                                "MM": "month",
                                "DD": "day",
                                "hh": "hour",
                            }
                        )
                    )
                    df.drop(columns=["YYYY", "MM", "DD", "hh"], inplace=True)
                except Exception as _e:
                    df["date"] = pd.to_datetime(
                        df[["#YY", "MM", "DD", "hh", "mm"]].rename(
                            columns={
                                "#YY": "year",
                                "MM": "month",
                                "DD": "day",
                                "hh": "hour",
                                "mm": "minute",
                            }
                        )
                    )
                    df.drop(columns=["#YY", "MM", "DD", "hh", "mm"], inplace=True)
                df.set_index("date", inplace=True)
                all_data.append(df)
            except FileNotFoundError:
                self.logger.error(
                    f"No wave spectra file found for buoy {buoy_id} year {year}"
                )

        if all_data:
            return pd.concat(all_data).sort_index()
        return None

    def _download_directional_spectra(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        **kwargs,
    ) -> str:
        """
        Download directional wave spectra coefficients.

        Parameters
        ----------
        data_type_config : dict
            The configuration for the data type.
        dataset_config : dict
            The configuration for the dataset.
        buoy_id : str
            The buoy ID.
        years : List[int]
            The years to download data for.

        Returns
        -------
        str
            The status message.
        """

        self.logger.info(
            f"Downloading directional spectra for buoy {buoy_id}, years {years}"
        )

        base_url = dataset_config["base_url"]
        coefficients = data_type_config["coefficients"]

        buoy_dir = os.path.join(
            self.base_path_to_download, "buoy_data", buoy_id, "directional_spectra"
        )
        if not self.check:
            os.makedirs(buoy_dir, exist_ok=True)

        downloaded_files = []

        for year in years:
            for coef, info in coefficients.items():
                filename = f"{buoy_id}{coef}{year}.txt.gz"
                url = f"{base_url}/{info['url_pattern'].format(buoy_id=buoy_id, year=year)}"

                if not self.check:
                    save_path = os.path.join(buoy_dir, filename)

                    try:
                        self.logger.debug(
                            f"Downloading {info['name']} data for {year}..."
                        )
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        # Save the compressed file
                        with open(save_path, "wb") as f:
                            shutil.copyfileobj(response.raw, f)

                        downloaded_files.append(save_path)
                        self.logger.info(f"Successfully downloaded {filename}")

                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"Error downloading {filename}: {e}")
                        continue

        return f"Downloaded {len(downloaded_files)} coefficient files"

    def read_directional_spectra(
        self, buoy_id: str, years: Union[int, List[int]]
    ) -> Tuple[Optional[pd.DataFrame], ...]:
        """
        Read directional spectra data for a specific buoy and year(s).

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        years : Union[int, List[int]]
            The year(s) to read data for. Can be a single year or a list of years.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], ...]
            Tuple containing DataFrames for alpha1, alpha2, r1, r2, and c11,
            or None for each if data not found
        """

        if isinstance(years, int):
            years = [years]

        results = {
            "alpha1": [],
            "alpha2": [],
            "r1": [],
            "r2": [],
            "c11": [],
        }

        for year in years:
            dir_path = os.path.join(
                self.base_path_to_download,
                "buoy_data",
                buoy_id,
                "directional_spectra",
            )
            files = {
                "alpha1": f"{buoy_id}d{year}.txt.gz",
                "alpha2": f"{buoy_id}i{year}.txt.gz",
                "r1": f"{buoy_id}j{year}.txt.gz",
                "r2": f"{buoy_id}k{year}.txt.gz",
                "c11": f"{buoy_id}w{year}.txt.gz",
            }

            for name, filename in files.items():
                file_path = os.path.join(dir_path, filename)
                try:
                    df = self._read_directional_file(file_path)
                    if df is not None:
                        results[name].append(df)
                except FileNotFoundError:
                    self.logger.error(
                        f"No {name} file found for buoy {buoy_id} year {year}"
                    )

        # Combine DataFrames for each coefficient if available
        final_results = {}
        for name, dfs in results.items():
            if dfs:
                final_results[name] = pd.concat(dfs).sort_index()
            else:
                final_results[name] = None

        return (
            final_results["alpha1"],
            final_results["alpha2"],
            final_results["r1"],
            final_results["r2"],
            final_results["c11"],
        )

    def _read_directional_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a directional spectra file and return DataFrame with datetime index.

        Parameters
        ----------
        file_path : Path
            Path to the file to read

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the directional spectra data, or None if data not found
        """

        self.logger.debug(f"Reading file: {file_path}")
        try:
            with gzip.open(file_path, "rt") as f:
                # Read header lines until we find the frequencies
                header_lines = []
                while True:
                    line = f.readline().strip()
                    if not line.startswith("#") and not line.startswith("YYYY"):
                        break
                    header_lines.append(line)

                # Parse frequencies
                header = " ".join(header_lines)
                try:
                    freqs = [float(x) for x in header.split()[5:]]
                    self.logger.debug(f"Found {len(freqs)} frequencies")
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Error parsing frequencies: {e}")
                    return None

                # Read data
                data = []
                dates = []
                # Process the first line
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        values = [float(x) for x in parts[5:]]
                        if len(values) == len(freqs):
                            dates.append(datetime(year, month, day, hour, minute))
                            data.append(values)
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error parsing line: {e}")

                # Read remaining lines
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            year, month, day, hour, minute = map(int, parts[:5])
                            values = [float(x) for x in parts[5:]]
                            if len(values) == len(freqs):
                                dates.append(datetime(year, month, day, hour, minute))
                                data.append(values)
                        except (ValueError, IndexError) as e:
                            self.logger.error(f"Error parsing line: {e}")
                            continue

                if not data:
                    self.logger.warning("No valid data points found in file")
                    return None

                df = pd.DataFrame(data, index=dates, columns=freqs)
                self.logger.debug(f"Created DataFrame with shape: {df.shape}")
                return df

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def _download_wind_forecast(
        self,
        data_type_config: dict,
        dataset_config: dict,
        date: str = None,
        region: List[float] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Download NOAA GFS wind forecast data.

        Parameters
        ----------
        data_type_config : dict
            The configuration for the data type.
        dataset_config : dict
            The configuration for the dataset.
        date : str, optional
            The date to download data for.

        Returns
        -------
        xr.Dataset
            The downloaded data.

        Notes
        -----
        - This will be DEPRECATED in the future.
        """

        if date is None:
            date = datetime.today().strftime("%Y%m%d")

        self.logger.info(f"Downloading wind forecast for date {date}")

        url_base = dataset_config["base_url"]
        dbn = "gfs_0p25_1hr"
        url = f"{url_base}/gfs{date}/{dbn}_00z"

        # File path for local storage
        forecast_dir = os.path.join(self.base_path_to_download, "wind_forecast")
        if not self.check:
            os.makedirs(forecast_dir, exist_ok=True)

        file_path = os.path.join(
            forecast_dir, f"{date}_{'_'.join(map(str, region))}.nc"
        )

        # Check if file exists
        if os.path.isfile(file_path):
            self.logger.info(
                f"File already exists: {file_path}. Loading from local storage."
            )
            data = xr.open_dataset(file_path)
        else:
            if self.check:
                self.logger.info(f"File would be downloaded to: {file_path}")
                return None

            self.logger.info(f"Downloading and cropping forecast data from: {url}")
            # Crop dataset
            data = xr.open_dataset(url)

            # Select only wind data
            variables = data_type_config["variables"]
            data_select = data[variables]

            self.logger.info(f"Storing local copy at: {file_path}")
            data_select.to_netcdf(file_path)
            data = data_select

        # Create output dataset with renamed variables
        output_vars = data_type_config["output_variables"]
        wind_data_forecast = xr.Dataset(
            {
                output_vars["u10"]: (
                    ("time", "lat", "lon"),
                    data[data_type_config["variables"][0]].values,
                ),
                output_vars["v10"]: (
                    ("time", "lat", "lon"),
                    data[data_type_config["variables"][1]].values,
                ),
            },
            coords={
                "time": data.time.values,
                "lat": data.lat.values,
                "lon": data.lon.values,
            },
        )
        wind_data_forecast["time"] = wind_data_forecast.time.dt.round("min")

        return wind_data_forecast

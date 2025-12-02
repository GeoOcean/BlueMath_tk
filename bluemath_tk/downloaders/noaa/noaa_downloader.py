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
from .._download_result import DownloadResult


def read_bulk_parameters(
    base_path: str, buoy_id: str, years: Union[int, List[int]]
) -> Optional[pd.DataFrame]:
    """
    Read bulk parameters for a specific buoy and year(s).

    Parameters
    ----------
    base_path : str
        Base path where the data is stored.
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
            base_path,
            "NDBC",
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
            print(f"No bulk parameters file found for buoy {buoy_id} year {year}")

    if all_data:
        return pd.concat(all_data, ignore_index=True).sort_values("datetime")
    return None


def read_wave_spectra(
    base_path: str, buoy_id: str, years: Union[int, List[int]]
) -> Optional[pd.DataFrame]:
    """
    Read wave spectra data for a specific buoy and year(s).

    Parameters
    ----------
    base_path : str
        Base path where the data is stored.
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
            base_path,
            "NDBC",
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
            except Exception:
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
            print(f"No wave spectra file found for buoy {buoy_id} year {year}")

    if all_data:
        return pd.concat(all_data).sort_index()
    return None


def _read_directional_file(file_path: Path) -> Optional[pd.DataFrame]:
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

    print(f"Reading file: {file_path}")
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
                print(f"Found {len(freqs)} frequencies")
            except (ValueError, IndexError) as e:
                print(f"Error parsing frequencies: {e}")
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
                    print(f"Error parsing line: {e}")

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
                        print(f"Error parsing line: {e}")
                        continue

            if not data:
                print("No valid data points found in file")
                return None

            df = pd.DataFrame(data, index=dates, columns=freqs)
            print(f"Created DataFrame with shape: {df.shape}")
            return df

    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def read_directional_spectra(
    base_path: str, buoy_id: str, years: Union[int, List[int]]
) -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Read directional spectra data for a specific buoy and year(s).

    Parameters
    ----------
    base_path : str
        Base path where the data is stored.
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
            base_path,
            "NDBC",
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
                df = _read_directional_file(file_path)
                if df is not None:
                    results[name].append(df)
            except FileNotFoundError:
                print(f"No {name} file found for buoy {buoy_id} year {year}")

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


class NOAADownloader(BaseDownloader):
    """
    This is the main class to download data from NOAA.

    Examples
    --------
    >>> downloader = NOAADownloader(
    ...     product="NDBC",
    ...     base_path_to_download="./noaa_data",
    ...     debug=True
    ... )
    >>> result = downloader.download_data(
    ...     data_type="bulk_parameters",
    ...     buoy_id="41001",
    ...     years=[2023],
    ...     dry_run=False
    ... )
    >>> print(result)
    """

    products_configs = {
        "NDBC": json.load(
            open(os.path.join(os.path.dirname(__file__), "NDBC", "NDBC_config.json"))
        )
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        debug: bool = True,
    ) -> None:
        """
        Initialize the NOAA downloader.

        Parameters
        ----------
        product : str
            The product to download data from. Currently only NDBC is supported.
        base_path_to_download : str
            The base path to download the data to.
        debug : bool, optional
            Whether to run in debug mode. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found.
        """

        super().__init__(
            product=product, base_path_to_download=base_path_to_download, debug=debug
        )

        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            available_products = list(self.products_configs.keys())
            raise ValueError(
                f"Product '{product}' not found. Available: {available_products}"
            )

        self.set_logger_name(
            f"NOAADownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        self.logger.info(f"---- NOAA DOWNLOADER INITIALIZED ({product}) ----")

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
    def data_types(self) -> dict:
        """
        Data types configuration dictionary.

        Returns
        -------
        dict
            Dictionary of available data types and their configurations.
        """
        return self.product_config["data_types"]

    def list_data_types(self) -> List[str]:
        """
        List all available data types for the product.

        Returns
        -------
        List[str]
            List of available data type names.
        """
        return list(self.data_types.keys())

    def _check_file_exists(
        self, file_path: str, result: DownloadResult, force: bool, dry_run: bool
    ) -> bool:
        """
        Check if file exists and handle accordingly.

        Parameters
        ----------
        file_path : str
            Path to the file to check.
        result : DownloadResult
            The download result to update.
        force : bool
            Whether to force re-download.
        dry_run : bool
            If True, only check files without downloading.

        Returns
        -------
        bool
            True if should skip download (file exists or dry_run mode), False otherwise.
        """

        if not force and os.path.exists(file_path):
            result.add_skipped(file_path, "File already exists")
            return True

        if dry_run:
            result.add_skipped(file_path, "File does not exist (dry run)")
            return True

        return False

    def download_data(self, dry_run: bool = True, *args, **kwargs) -> DownloadResult:
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

        if self.product == "NDBC":
            return self.download_data_ndbc(dry_run=dry_run, *args, **kwargs)
        else:
            raise ValueError(f"Download for product {self.product} not supported")

    def download_data_ndbc(
        self, data_type: str, dry_run: bool = True, **kwargs
    ) -> DownloadResult:
        """
        Download data for the NDBC product.

        Downloads NDBC buoy data or forecast data based on the specified data type.
        Files are saved to: base_path_to_download/product/dataset/...

        Parameters
        ----------
        data_type : str
            The data type to download. Available types:
            - 'bulk_parameters': Standard meteorological data
            - 'wave_spectra': Wave spectral density data
            - 'directional_spectra': Directional wave spectra coefficients
            - 'wind_forecast': GFS wind forecast data
        dry_run : bool, optional
            If True, only check what would be downloaded without actually downloading.
            Default is True.
        **kwargs
            Additional keyword arguments specific to each data type:
            - For bulk_parameters, wave_spectra, directional_spectra: buoy_id, years, force
            - For wind_forecast: date, region, force

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.

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
        dataset_config = self.product_config["datasets"][data_type_config["dataset"]]

        if dry_run:
            self.logger.info(f"DRY RUN: Checking files for {data_type}")

        if data_type == "bulk_parameters":
            result = self._download_bulk_parameters(
                data_type_config, dataset_config, dry_run=dry_run, **kwargs
            )
        elif data_type == "wave_spectra":
            result = self._download_wave_spectra(
                data_type_config, dataset_config, dry_run=dry_run, **kwargs
            )
        elif data_type == "directional_spectra":
            result = self._download_directional_spectra(
                data_type_config, dataset_config, dry_run=dry_run, **kwargs
            )
        elif data_type == "wind_forecast":
            result = self._download_wind_forecast(
                data_type_config, dataset_config, dry_run=dry_run, **kwargs
            )
        else:
            raise ValueError(f"Download for data type {data_type} not implemented")

        return self.finalize_download_result(result)

    def _download_bulk_parameters(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
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
        force : bool, optional
            Whether to force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Download result with information about downloaded, skipped, and error files.
        """

        self.logger.info(
            f"Downloading bulk parameters for buoy {buoy_id}, years {years}"
        )

        result = self.create_download_result()
        base_url = dataset_config["base_url"]
        dataset_name = data_type_config["dataset"]

        try:
            # Determine output file path: base_path/product/dataset/buoy_id/filename.csv
            buoy_dir = os.path.join(
                self.base_path_to_download, self.product, dataset_name, buoy_id
            )
            output_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_bulk_parameters.csv")

            # Check if file exists
            if self._check_file_exists(output_file, result, force, dry_run):
                return self.finalize_download_result(result)

            # Prepare download tasks
            download_tasks = []
            for year in years:
                urls = [
                    f"{base_url}/{data_type_config['url_pattern'].format(buoy_id=buoy_id, year=year)}"
                ]
                for fallback in data_type_config.get("fallback_urls", []):
                    urls.append(
                        f"{base_url}/{fallback.format(buoy_id=buoy_id, year=year)}"
                    )

                download_tasks.append(
                    {
                        "urls": urls,
                        "columns": data_type_config["columns"],
                        "year": year,
                        "buoy_id": buoy_id,
                    }
                )

            if dry_run:
                # In dry run mode, just mark what would be downloaded
                for task in download_tasks:
                    result.add_skipped(
                        output_file,
                        f"Would download year {task['year']} (dry run)",
                    )
                return self.finalize_download_result(result)

            # Execute downloads sequentially
            all_data = []
            for task in download_tasks:
                try:
                    df = self._download_single_year_bulk(task["urls"], task["columns"])
                    if df is not None:
                        all_data.append(df)
                        self.logger.info(
                            f"Buoy {buoy_id}: Data found for year {task['year']}"
                        )
                    else:
                        self.logger.warning(
                            f"Buoy {buoy_id}: No data available for year {task['year']}"
                        )
                        result.add_error(
                            output_file,
                            Exception(f"No data available for year {task['year']}"),
                        )
                except Exception as e:
                    self.logger.error(f"Error downloading year {task['year']}: {e}")
                    result.add_error(output_file, e)

            if all_data:
                # Combine all years
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.sort_values(["YYYY", "MM", "DD", "hh"])

                # Save to CSV
                os.makedirs(buoy_dir, exist_ok=True)
                combined_df.to_csv(output_file, index=False)
                self.logger.info(f"Data saved to {output_file}")
                result.add_downloaded(output_file)
            else:
                self.logger.error(f"No data found for buoy {buoy_id}")
                result.add_error(
                    output_file,
                    Exception(f"No data found for buoy {buoy_id}"),
                )
        except Exception as e:
            result.add_error(output_file, e)
            self.logger.error(f"Error processing data for buoy {buoy_id}: {e}")

        return self.finalize_download_result(result)

    def _download_single_year_bulk(
        self,
        urls: List[str],
        columns: List[str],
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse bulk parameters for a single year.

        Attempts to download from the primary URL, and if that fails, tries fallback URLs.
        Handles different data formats (pre-2012 and post-2012) and validates dates.

        Parameters
        ----------
        urls : List[str]
            List of URLs to try downloading from (primary URL first, then fallbacks).
        columns : List[str]
            List of column names for the DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the downloaded and parsed data, or None if download fails.
        """

        for url in urls:
            try:
                # Download the file
                response = requests.get(url, timeout=30)
                response.raise_for_status()
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

    def _download_wave_spectra(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download wave spectra data for a specific buoy.

        Downloads wave spectral density data for each specified year. Files are saved to:
        base_path_to_download/product/dataset/buoy_id/wave_spectra/buoy_{buoy_id}_spectra_{year}.csv

        Parameters
        ----------
        data_type_config : dict
            Configuration for the data type.
        dataset_config : dict
            Configuration for the dataset.
        buoy_id : str
            The buoy ID.
        years : List[int]
            List of years to download data for.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
        """

        self.logger.info(f"Downloading wave spectra for buoy {buoy_id}, years {years}")

        result = self.create_download_result()
        base_url = dataset_config["base_url"]
        dataset_name = data_type_config["dataset"]
        buoy_dir = os.path.join(
            self.base_path_to_download,
            self.product,
            dataset_name,
            buoy_id,
            "wave_spectra",
        )

        if not dry_run:
            os.makedirs(buoy_dir, exist_ok=True)

        for year in years:
            url = f"{base_url}/{data_type_config['url_pattern'].format(buoy_id=buoy_id, year=year)}"
            output_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_spectra_{year}.csv")

            # Check if file exists
            if self._check_file_exists(output_file, result, force, dry_run):
                continue

            if dry_run:
                result.add_skipped(output_file, f"Would download year {year} (dry run)")
                continue

            try:
                # Download and read the data
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Read the data
                df = pd.read_csv(
                    io.BytesIO(response.content),
                    compression="gzip",
                    sep=r"\s+",
                    na_values=["MM", "99.00", "999.0"],
                )

                # Skip if empty or invalid data
                if df.empty or len(df.columns) < 5:
                    self.logger.warning(f"No valid data for {buoy_id} - {year}")
                    result.add_error(
                        output_file,
                        Exception(f"No valid data for {buoy_id} - {year}"),
                        context={"year": year},
                    )
                    continue

                # Save the data
                df.to_csv(output_file, index=False)
                result.add_downloaded(output_file)
                self.logger.info(f"Successfully saved data for {buoy_id} - {year}")

            except Exception as e:
                self.logger.warning(f"No data found for: {buoy_id} - {year}: {e}")
                result.add_error(output_file, e, context={"year": year})
                continue

        return result

    def _download_directional_spectra(
        self,
        data_type_config: dict,
        dataset_config: dict,
        buoy_id: str,
        years: List[int],
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download directional wave spectra coefficients.

        Downloads Fourier coefficients (alpha1, alpha2, r1, r2, c11) for directional wave spectra.
        Files are saved to:
        base_path_to_download/product/dataset/buoy_id/directional_spectra/{buoy_id}{coef}{year}.txt.gz

        Parameters
        ----------
        data_type_config : dict
            Configuration for the data type.
        dataset_config : dict
            Configuration for the dataset.
        buoy_id : str
            The buoy ID.
        years : List[int]
            List of years to download data for.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
        """

        self.logger.info(
            f"Downloading directional spectra for buoy {buoy_id}, years {years}"
        )

        result = self.create_download_result()
        base_url = dataset_config["base_url"]
        coefficients = data_type_config["coefficients"]
        dataset_name = data_type_config["dataset"]

        buoy_dir = os.path.join(
            self.base_path_to_download,
            self.product,
            dataset_name,
            buoy_id,
            "directional_spectra",
        )
        if not dry_run:
            os.makedirs(buoy_dir, exist_ok=True)

        for year in years:
            for coef, info in coefficients.items():
                filename = f"{buoy_id}{coef}{year}.txt.gz"
                url = f"{base_url}/{info['url_pattern'].format(buoy_id=buoy_id, year=year)}"
                save_path = os.path.join(buoy_dir, filename)

                # Check if file exists
                if self._check_file_exists(save_path, result, force, dry_run):
                    continue

                if dry_run:
                    result.add_skipped(
                        save_path,
                        f"Would download {info['name']} for year {year} (dry run)",
                    )
                    continue

                try:
                    self.logger.debug(f"Downloading {info['name']} data for {year}...")

                    # Download the file
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()

                    # Save the compressed file
                    with open(save_path, "wb") as f:
                        shutil.copyfileobj(response.raw, f)

                    result.add_downloaded(save_path)
                    self.logger.info(f"Successfully downloaded {filename}")

                except Exception as e:
                    self.logger.warning(f"Error downloading {filename}: {e}")
                    result.add_error(save_path, e)
                    continue

        return self.finalize_download_result(result)

    def _download_wind_forecast(
        self,
        data_type_config: dict,
        dataset_config: dict,
        date: str = None,
        region: List[float] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download NOAA GFS wind forecast data.

        Downloads and crops GFS wind forecast data for a specific date and region.
        Files are saved to:
        base_path_to_download/product/dataset/{date}_{region}.nc

        Parameters
        ----------
        data_type_config : dict
            Configuration for the data type.
        dataset_config : dict
            Configuration for the dataset.
        date : str, optional
            Date to download data for (format: "YYYYMMDD"). If None, uses today's date.
            Default is None.
        region : List[float], optional
            Geographic region coordinates. Default is None.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.

        Notes
        -----
        This method will be DEPRECATED in the future.
        """

        if date is None:
            date = datetime.today().strftime("%Y%m%d")

        self.logger.info(f"Downloading wind forecast for date {date}")

        result = self.create_download_result()
        url_base = dataset_config["base_url"]
        dataset_name = data_type_config["dataset"]
        dbn = "gfs_0p25_1hr"
        url = f"{url_base}/gfs{date}/{dbn}_00z"

        # File path for local storage: base_path/product/dataset/filename.nc
        forecast_dir = os.path.join(
            self.base_path_to_download, self.product, dataset_name
        )
        if not dry_run:
            os.makedirs(forecast_dir, exist_ok=True)

        file_path = os.path.join(
            forecast_dir, f"{date}_{'_'.join(map(str, region))}.nc"
        )

        # Check if file exists
        if self._check_file_exists(file_path, result, force, dry_run):
            return result

        if dry_run:
            result.add_skipped(
                file_path, f"Would download wind forecast for {date} (dry run)"
            )
            return result

        try:
            self.logger.info(f"Downloading and cropping forecast data from: {url}")
            # Open dataset from URL
            data = xr.open_dataset(url)

            # Select only wind data
            variables = data_type_config["variables"]
            data_select = data[variables]

            self.logger.info(f"Storing local copy at: {file_path}")
            data_select.to_netcdf(file_path)
            result.add_downloaded(file_path)

        except Exception as e:
            self.logger.error(f"Error downloading wind forecast: {e}")
            result.add_error(file_path, e)

        return self.finalize_download_result(result)

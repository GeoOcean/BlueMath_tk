"""
Downloader for the CSIRO wave hindcasts, served by the CSIRO THREDDS server.

Products (configs in ``csiro/<PRODUCT>/<PRODUCT>_config.json``):

- WHACS: WAVEWATCH III hindcast forced by ERA5, global SMC grid, 1979-2023.
- CAWCR: WAVEWATCH III hindcast, global gridded fields and station spectra,
  1979-present.

Full files are downloaded through the THREDDS fileServer and time series at
given locations are extracted through OPeNDAP.
"""

import calendar
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .._base_downloaders import BaseDownloader, load_config
from .._download_result import DownloadResult
from .._transfer import (
    PART_SUFFIX,
    RemoteFileNotFound,
    http_download,
    make_session,
    remote_size,
)

_HERE = os.path.dirname(__file__)
EARTH_RADIUS_KM = 6371.0
MAX_DISTANCE_WARNING_KM = 25.0


def nearest_point_index(
    lons: np.ndarray, lats: np.ndarray, lon: float, lat: float
) -> Tuple[int, float]:
    """
    Index of the point closest to (lon, lat), using the haversine distance.

    Longitudes may use any convention (-180/180 or 0/360).

    Parameters
    ----------
    lons, lats : np.ndarray
        Longitudes and latitudes of the candidate points, in degrees.
    lon, lat : float
        Target longitude and latitude, in degrees.

    Returns
    -------
    Tuple[int, float]
        Flat index of the nearest point and its distance in km.
    """

    lons = np.radians(np.asarray(lons, dtype=float)).ravel()
    lats = np.radians(np.asarray(lats, dtype=float)).ravel()
    lon0, lat0 = np.radians(lon), np.radians(lat)
    a = (
        np.sin((lats - lat0) / 2) ** 2
        + np.cos(lat0) * np.cos(lats) * np.sin((lons - lon0) / 2) ** 2
    )
    idx = int(np.nanargmin(a))
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a[idx]))

    return idx, float(distance)


class CSIRODownloader(BaseDownloader):
    """
    Downloader for the CSIRO wave hindcasts served by the CSIRO THREDDS server.

    Two ways of getting data are available:

    - ``download_data``: full monthly files through the THREDDS fileServer.
      Transfers are resumable, checked against the remote file size and run in
      parallel (``max_workers``).
    - ``download_point``: time series at one or more locations through OPeNDAP,
      without downloading the full files. One file is written per month and,
      optionally, all months are combined into a single file.

    Files are saved to base_path_to_download/<local_dir of the dataset>, so with
    base_path_to_download="/lustre/geocean/DATA/CSIRO" the existing GeoOcean
    layout is reused and already downloaded files are skipped.

    Examples
    --------
    >>> downloader = CSIRODownloader(
    ...     product="WHACS",
    ...     base_path_to_download="/lustre/geocean/DATA/CSIRO",
    ... )
    >>> result = downloader.download_data(
    ...     dataset="global-smc",
    ...     variables=["phs1", "ptp1"],
    ...     start="2000-01",
    ...     end="2000-12",
    ...     dry_run=False,
    ... )
    >>> result = downloader.download_point(
    ...     dataset="global-smc",
    ...     lon=-4.0,
    ...     lat=44.0,
    ...     variables=["hs", "t02"],
    ...     dry_run=False,
    ... )
    """

    products_configs = {
        "WHACS": load_config(_HERE, "WHACS", "WHACS_config.json"),
        "CAWCR": load_config(_HERE, "CAWCR", "CAWCR_config.json"),
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        max_workers: int = 2,
        debug: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the CSIRODownloader.

        Parameters
        ----------
        product : str
            The product to download data from ("WHACS" or "CAWCR").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        max_workers : int, optional
            Number of files downloaded in parallel by download_data. Default is 2.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        **kwargs
            Passed to BaseDownloader (e.g., retries, retry_backoff).

        Raises
        ------
        ValueError
            If the product configuration is not found.
        """

        super().__init__(
            product=product,
            base_path_to_download=base_path_to_download,
            debug=debug,
            **kwargs,
        )

        self.max_workers = max_workers
        self._local = threading.local()
        self._grid_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._exclude_attributes.extend(["_local", "_grid_cache"])

        self.set_logger_name(
            f"CSIRODownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        self.logger.info(f"---- CSIRO DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def session(self) -> requests.Session:
        """
        HTTP session of the current thread.

        Returns
        -------
        requests.Session
            Session with retries for connection errors and 5xx responses.
        """
        if not hasattr(self._local, "session"):
            self._local.session = make_session()
        return self._local.session

    def list_variables(self, dataset: str) -> List[str]:
        """
        List the variables of a dataset with one file per variable.

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns
        -------
        List[str]
            List of variable names (empty if all variables are in the same file).
        """

        return list(self._get_dataset_config(dataset).get("variables", {}).keys())

    def get_months(
        self, dataset: str, start: str = None, end: str = None
    ) -> pd.PeriodIndex:
        """
        Months to process, limited to the dataset availability.

        Parameters
        ----------
        dataset : str
            The dataset name.
        start : str, optional
            First month ("YYYY-MM"). Default is the dataset start.
        end : str, optional
            Last month ("YYYY-MM"). Default is the dataset end or, for datasets
            that are still updated, the current month.

        Returns
        -------
        pd.PeriodIndex
            Monthly periods.
        """

        dataset_config = self._get_dataset_config(dataset)
        first = pd.Period(dataset_config["start"], freq="M")
        last = (
            pd.Period(dataset_config["end"], freq="M")
            if dataset_config.get("end")
            else pd.Period.now(freq="M")
        )
        start = max(pd.Period(start, freq="M"), first) if start else first
        end = min(pd.Period(end, freq="M"), last) if end else last

        return pd.period_range(start, end, freq="M")

    def get_url(
        self,
        dataset: str,
        period: pd.Period,
        variable: str = None,
        service: str = "fileserver",
    ) -> str:
        """
        URL of a monthly file.

        Parameters
        ----------
        dataset : str
            The dataset name.
        period : pd.Period
            The month.
        variable : str, optional
            The variable, for datasets with one file per variable.
        service : str, optional
            "fileserver" (download) or "opendap" (remote access). Default is "fileserver".

        Returns
        -------
        str
            The URL.
        """

        dataset_config = self._get_dataset_config(dataset)
        base_url = self.product_config[f"{service}_url"]
        template_args = self._template_args(period, variable)
        remote_dir = dataset_config["remote_dir"].format(**template_args)
        filename = dataset_config["filename"].format(**template_args)

        return f"{base_url}{remote_dir}{filename}"

    def get_local_path(
        self, dataset: str, period: pd.Period, variable: str = None
    ) -> str:
        """
        Local path of a monthly file.

        Parameters
        ----------
        dataset : str
            The dataset name.
        period : pd.Period
            The month.
        variable : str, optional
            The variable, for datasets with one file per variable.

        Returns
        -------
        str
            The local path.
        """

        dataset_config = self._get_dataset_config(dataset)
        template_args = self._template_args(period, variable)
        template_args["era"] = self._get_era(dataset_config, period)
        local_dir = dataset_config["local_dir"].format(**template_args)
        filename = dataset_config["filename"].format(**template_args)

        return os.path.join(self.base_path_to_download, local_dir, filename)

    def download_data(
        self,
        dataset: str,
        variables: List[str] = None,
        start: str = None,
        end: str = None,
        force: bool = False,
        verify_existing: bool = True,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Download full monthly files.

        Months that are not (yet) on the server are reported as skipped, so the
        same call can be repeated to keep a dataset up to date.

        Parameters
        ----------
        dataset : str
            The dataset to download. Use list_datasets() to see available datasets.
        variables : List[str], optional
            Variables to download, required for datasets with one file per
            variable (see list_variables). Ignored otherwise.
        start : str, optional
            First month ("YYYY-MM"). Default is the dataset start.
        end : str, optional
            Last month ("YYYY-MM"). Default is the last available month.
        force : bool, optional
            Download again even if the file exists. Default is False.
        verify_existing : bool, optional
            Compare the size of existing files with the remote size and download
            them again if they differ (e.g. interrupted transfers). Default is True.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
            The remote size of each checked file is in metadata["remote_sizes"].
        """

        dataset_config = self._get_dataset_config(dataset)
        file_variables = self._check_variables(dataset, dataset_config, variables)

        tasks = [
            (
                self.get_url(dataset, period, variable),
                self.get_local_path(dataset, period, variable),
            )
            for variable in file_variables
            for period in self.get_months(dataset, start, end)
        ]
        self.logger.info(f"Checking {len(tasks)} files of {self.product}/{dataset}")

        result = self.create_download_result()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._download_file, url, path, force, verify_existing, dry_run
                )
                for url, path in tasks
            ]
            for future in futures:
                result.merge(future.result())

        return self.finalize_download_result(result)

    def download_point(
        self,
        dataset: str,
        lon: Union[float, Sequence[float]],
        lat: Union[float, Sequence[float]],
        variables: List[str] = None,
        start: str = None,
        end: str = None,
        output_dir: str = None,
        name: str = None,
        combine: bool = True,
        force: bool = False,
        dry_run: bool = True,
    ) -> DownloadResult:
        """
        Extract time series at the nearest model point(s) through OPeNDAP.

        One file per month is saved to output_dir/{name}_{YYYYMM}.nc. If combine
        is True, all available months are also merged into
        output_dir/{name}_{dataset}_{start}_{end}.nc.

        Parameters
        ----------
        dataset : str
            The dataset to extract from.
        lon, lat : float or Sequence[float]
            Target location(s), in degrees.
        variables : List[str], optional
            Variables to extract. Required for datasets with one file per variable.
            For other datasets, None extracts all variables.
        start : str, optional
            First month ("YYYY-MM"). Default is the dataset start.
        end : str, optional
            Last month ("YYYY-MM"). Default is the last available month.
        output_dir : str, optional
            Output directory. Default is
            base_path_to_download/product/dataset/points/name.
        name : str, optional
            Name of the extraction, used for file names. Required for several
            locations. Default is "lon{lon}_lat{lat}".
        combine : bool, optional
            Merge all monthly files into a single file. Default is True.
        force : bool, optional
            Extract again even if the monthly file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be extracted. Default is True.

        Returns
        -------
        DownloadResult
            Result with the monthly files. The combined file, if any, is in
            result.metadata["combined_file"].
        """

        dataset_config = self._get_dataset_config(dataset)
        lons = np.atleast_1d(np.asarray(lon, dtype=float))
        lats = np.atleast_1d(np.asarray(lat, dtype=float))
        if lons.shape != lats.shape:
            raise ValueError("lon and lat must have the same length")
        if dataset_config["file_per_variable"]:
            self._check_variables(dataset, dataset_config, variables)
        if name is None:
            if len(lons) > 1:
                raise ValueError("name is required when extracting several points")
            name = f"lon{lons[0]:.3f}_lat{lats[0]:.3f}"
        if output_dir is None:
            output_dir = os.path.join(
                self.base_path_to_download, self.product, dataset, "points", name
            )

        months = self.get_months(dataset, start, end)
        self.logger.info(
            f"Extracting {len(months)} months of {self.product}/{dataset} at {name}"
        )

        result = self.create_download_result()
        monthly_files = []
        for period in months:
            output_file = os.path.join(
                output_dir, f"{name}_{period.year}{period.month:02d}.nc"
            )
            monthly_files.append(output_file)

            def fetch(part_file: str, period: pd.Period = period) -> None:
                point_ds = self._extract_point_month(
                    dataset, dataset_config, period, variables, lons, lats
                )
                point_ds.to_netcdf(part_file)

            self.fetch_file(output_file, fetch, result, force=force, dry_run=dry_run)

        available_files = [f for f in monthly_files if os.path.exists(f)]
        if combine and not dry_run and available_files:
            combined_file = os.path.join(
                output_dir,
                f"{name}_{dataset}_{months[0].strftime('%Y%m')}_{months[-1].strftime('%Y%m')}.nc",
            )
            try:
                self.combine_point_files(available_files, combined_file)
                result.metadata["combined_file"] = combined_file
                self.logger.info(f"Combined file: {combined_file}")
            except Exception as e:
                self.logger.error(f"Error combining point files: {e}")
                result.add_error(combined_file, e)

        return self.finalize_download_result(result)

    @staticmethod
    def combine_point_files(files: List[str], output_file: str) -> None:
        """
        Concatenate monthly point files along time into a single NetCDF file.

        Parameters
        ----------
        files : List[str]
            Monthly files written by download_point.
        output_file : str
            Path of the combined file. It is overwritten if it exists.
        """

        part_file = output_file + PART_SUFFIX
        with xr.open_mfdataset(
            sorted(files),
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="different",
        ) as ds:
            ds.drop_duplicates("time").to_netcdf(part_file)
        os.replace(part_file, output_file)

    def _get_dataset_config(self, dataset: str) -> dict:
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {self.list_datasets()}"
            )
        return self.product_config["datasets"][dataset]

    def _check_variables(
        self, dataset: str, dataset_config: dict, variables: Optional[List[str]]
    ) -> List[Optional[str]]:
        """
        Variables that define the files to process ([None] if one file holds all).
        """

        if not dataset_config["file_per_variable"]:
            return [None]
        if not variables:
            raise ValueError(
                f"variables is required for {self.product}/{dataset}. "
                f"Available: {self.list_variables(dataset)}"
            )
        unknown = set(variables) - set(self.list_variables(dataset))
        if unknown:
            raise ValueError(
                f"Unknown variables {sorted(unknown)} for {self.product}/{dataset}. "
                f"Available: {self.list_variables(dataset)}"
            )
        return list(variables)

    @staticmethod
    def _template_args(period: pd.Period, variable: Optional[str]) -> dict:
        return {
            "year": period.year,
            "month": period.month,
            "last_day": calendar.monthrange(period.year, period.month)[1],
            "variable": variable,
        }

    @staticmethod
    def _get_era(dataset_config: dict, period: pd.Period) -> Optional[str]:
        """
        Name of the era (file format period) a month belongs to, if any.
        """

        for era, (era_start, era_end) in dataset_config.get("eras", {}).items():
            if period >= pd.Period(era_start, freq="M") and (
                era_end is None or period <= pd.Period(era_end, freq="M")
            ):
                return era
        return None

    def _download_file(
        self,
        url: str,
        output_file: str,
        force: bool,
        verify_existing: bool,
        dry_run: bool,
    ) -> DownloadResult:
        """
        Download a single file, checking the remote size first.
        """

        result = DownloadResult()
        exists = os.path.exists(output_file)
        if exists and not force and not verify_existing:
            result.add_skipped(output_file, "Already downloaded")
            return result

        try:
            size = remote_size(self.session, url)
        except RemoteFileNotFound:
            reason = "Not available on server"
            result.add_skipped(
                output_file, reason + (" (local copy kept)" if exists else "")
            )
            return result
        except Exception as e:
            self.logger.error(f"Error checking {url}: {e}")
            result.add_error(output_file, e)
            return result

        if size is not None:
            result.metadata.setdefault("remote_sizes", {})[output_file] = size

        if exists and not force:
            local_size = os.path.getsize(output_file)
            if size is None or local_size == size:
                result.add_skipped(output_file, "Already downloaded")
                return result
            message = f"Local size {local_size} differs from remote size {size}"
            if dry_run:
                result.add_skipped(
                    output_file, f"{message}, would download again (dry run)"
                )
                return result
            self.logger.warning(f"{message}, downloading again: {output_file}")
            force = True

        def fetch(part_file: str) -> None:
            http_download(self.session, url, part_file, expected_size=size, resume=True)

        self.fetch_file(
            output_file, fetch, result, force=force, dry_run=dry_run, resumable=True
        )

        return result

    def _extract_point_month(
        self,
        dataset: str,
        dataset_config: dict,
        period: pd.Period,
        variables: Optional[List[str]],
        lons: np.ndarray,
        lats: np.ndarray,
    ) -> xr.Dataset:
        """
        Extract one month at the nearest point(s) through OPeNDAP.
        """

        rename = dataset_config.get("rename", {})
        file_variables = variables if dataset_config["file_per_variable"] else [None]

        pieces = []
        for variable in file_variables:
            # OPeNDAP errors for missing files are not explicit, so check first
            remote_size(self.session, self.get_url(dataset, period, variable))
            url = self.get_url(dataset, period, variable, service="opendap")
            with xr.open_dataset(
                url, drop_variables=dataset_config.get("drop_variables")
            ) as ds:
                ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables})
                ds = ds.set_coords(
                    [c for c in ("longitude", "latitude") if c in ds.data_vars]
                )
                point_ds = self._select_points(dataset, dataset_config, ds, lons, lats)
                if variable is not None:
                    point_ds = point_ds[[variable]]
                elif variables:
                    missing = set(variables) - set(point_ds.data_vars)
                    if missing:
                        raise ValueError(f"Variables {sorted(missing)} not in {url}")
                    point_ds = point_ds[variables]
                pieces.append(point_ds.load())

        point_ds = xr.merge(pieces, compat="override", combine_attrs="override")
        # Monthly files also hold the first time step of the next month
        in_period = point_ds.time.to_index().to_period("M") == period
        point_ds = point_ds.isel(time=np.flatnonzero(in_period))
        point_ds.attrs.update(
            {
                "source": f"CSIRO {self.product} {dataset}",
                "target_longitude": lons.tolist(),
                "target_latitude": lats.tolist(),
            }
        )

        return point_ds

    def _select_points(
        self,
        dataset: str,
        dataset_config: dict,
        ds: xr.Dataset,
        lons: np.ndarray,
        lats: np.ndarray,
    ) -> xr.Dataset:
        """
        Lazily select the nearest point(s) of a dataset.

        Datasets with a point dimension (stations, SMC seapoints) are indexed
        along it. Regular grids are indexed along latitude and longitude.
        """

        point_dim = dataset_config.get("point_dim")
        lon_values, lat_values = self._get_grid(dataset, ds, point_dim)

        nearest = [
            nearest_point_index(lon_values, lat_values, lon, lat)
            for lon, lat in zip(lons, lats)
        ]
        distances = [distance for _, distance in nearest]
        for (lon, lat), distance in zip(zip(lons, lats), distances):
            message = f"Nearest point to ({lon}, {lat}) is {distance:.1f} km away"
            if distance > MAX_DISTANCE_WARNING_KM:
                self.logger.warning(message)
            else:
                self.logger.debug(message)

        if point_dim is not None:
            indices = np.array([idx for idx, _ in nearest])
            if len(indices) == 1:
                selected = ds.isel({point_dim: int(indices[0])})
            else:
                unique, inverse = np.unique(indices, return_inverse=True)
                selected = (
                    ds.isel({point_dim: unique}).load().isel({point_dim: inverse})
                )
        else:
            lat_idx, lon_idx = np.unravel_index(
                [idx for idx, _ in nearest], lon_values.shape
            )
            if len(lat_idx) == 1:
                selected = ds.isel(latitude=int(lat_idx[0]), longitude=int(lon_idx[0]))
            else:
                selected = ds.isel(
                    latitude=xr.DataArray(lat_idx, dims="point"),
                    longitude=xr.DataArray(lon_idx, dims="point"),
                )

        return selected.assign_attrs(nearest_point_distance_km=distances)

    def _get_grid(
        self, dataset: str, ds: xr.Dataset, point_dim: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Longitudes and latitudes of all points, cached when they do not change.
        """

        longitude, latitude = ds["longitude"], ds["latitude"]

        # Station positions stored per time step: read the first one, no caching
        if "time" in longitude.dims:
            return (
                longitude.isel(time=0).values,
                latitude.isel(time=0).values,
            )

        key = (dataset, longitude.size, latitude.size)
        if key not in self._grid_cache:
            if point_dim is None:
                lon_grid, lat_grid = np.meshgrid(longitude.values, latitude.values)
                self._grid_cache[key] = (lon_grid, lat_grid)
            else:
                self._grid_cache[key] = (longitude.values, latitude.values)

        return self._grid_cache[key]

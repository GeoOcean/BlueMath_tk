import ftplib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import xarray as xr

from .._base_downloaders import BaseDownloader
from .._download_result import DownloadResult


class AvisoDownloader(BaseDownloader):
    """
    Simple downloader for AVISO SWOT L3 Expert data.

    Uses configuration from SWOT_config.json to handle product details.
    Users only need to specify area and variables - the downloader handles everything else!

    Attributes
    ----------
    username : str
        AVISO FTP username
    password : str
        AVISO FTP password
    config : dict
        Product configuration loaded from SWOT_config.json
    dataset_config : dict
        Dataset-specific configuration

    Examples
    --------
    >>> from bluemath_tk.downloaders.aviso.aviso_downloader import AvisoDownloader
    >>>
    >>> downloader = AvisoDownloader(
    ...     base_path_to_download="./swot_data",
    ...     username="your_username",
    ...     password="your_password"
    ... )
    >>>
    >>> # List available variables
    >>> variables = downloader.list_variables()
    >>> print(variables)
    >>>
    >>> # Get variable information
    >>> info = downloader.get_variable_info('ssha_filtered')
    >>> print(info['long_name'])
    >>>
    >>> # Download data - just area and variables!
    >>> result = downloader.download_data(
    ...     variables=['ssha_filtered', 'time'],
    ...     lon_range=(-15, 40),  # Mediterranean
    ...     lat_range=(25, 50)
    ... )
    >>>
    >>> print(result)
    >>> # DownloadResult with all subset files ready to use
    """

    def __init__(
        self,
        base_path_to_download: str,
        username: str,
        password: str,
        debug: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the AvisoDownloader.

        Parameters
        ----------
        base_path_to_download : str
            Base path where downloaded files will be stored.
        username : str
            AVISO FTP username.
        password : str
            AVISO FTP password.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        max_retries : int, optional
            Maximum number of retry attempts. Default is 3.
        retry_delay : float, optional
            Initial retry delay in seconds. Default is 1.0.
        retry_backoff : float, optional
            Exponential backoff multiplier. Default is 2.0.
        show_progress : bool, optional
            Whether to show progress bars. Default is True.
        """

        super().__init__(
            base_path_to_download=base_path_to_download,
            debug=debug,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            show_progress=show_progress,
        )

        self._username = username
        self._password = password

        # Load config
        config_path = os.path.join(
            os.path.dirname(__file__), "SWOT", "SWOT_config.json"
        )
        self._config = json.load(open(config_path))

        # Get dataset config (default to swot-l3-expert for now)
        self._dataset_config = self._config["datasets"]["swot-l3-expert"]
        self._ftp_server = self._dataset_config["ftp_server"]
        self._ftp_base_path = self._dataset_config["ftp_base_path"]
        self._level = self._dataset_config["level"]
        self._variant = self._dataset_config["variant"]

        self.logger.info("---- AVISO DOWNLOADER INITIALIZED ----")

    @property
    def username(self) -> str:
        """AVISO FTP username."""
        return self._username

    @property
    def password(self) -> str:
        """AVISO FTP password."""
        return self._password

    @property
    def ftp_server(self) -> str:
        """FTP server address."""
        return self._ftp_server

    @property
    def config(self) -> dict:
        """Product configuration."""
        return self._config

    @property
    def dataset_config(self) -> dict:
        """Dataset-specific configuration."""
        return self._dataset_config

    def list_variables(self) -> List[str]:
        """
        List all available variables from the config.

        Returns
        -------
        List[str]
            List of variable names available for download.

        Examples
        --------
        >>> variables = downloader.list_variables()
        >>> print(variables)
        ['ssha_filtered', 'time', 'longitude', 'latitude', ...]
        """

        return list(self._config["variables"].keys())

    def get_variable_info(self, variable: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a variable.

        Parameters
        ----------
        variable : str
            Variable name

        Returns
        -------
        Dict[str, Any]
            Variable information (name, long_name, units, etc.)

        Examples
        --------
        >>> info = downloader.get_variable_info('ssha_filtered')
        >>> print(info['long_name'])
        Filtered Sea Surface Height Anomaly
        """

        return self._config["variables"].get(variable)

    def download_data(
        self,
        variables: List[str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download SWOT data for a geographical area.

        Automatically finds all cycles and passes that intersect the area.
        Downloads and subsets files automatically.

        Parameters
        ----------
        variables : List[str]
            Variables to download (e.g., ['ssha_filtered', 'time'])
            Use list_variables() to see available variables.
        lon_range : Tuple[float, float]
            Longitude range (min, max) in degrees
        lat_range : Tuple[float, float]
            Latitude range (min, max) in degrees
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Result with subset files ready to use. All files are already
            subset to the specified area.

        Examples
        --------
        >>> result = downloader.download_data(
        ...     variables=['ssha_filtered', 'time'],
        ...     lon_range=(-15, 40),  # Mediterranean
        ...     lat_range=(25, 50)
        ... )
        >>> print(f"Downloaded {len(result.downloaded_files)} files")
        >>> print(f"Duration: {result.duration_seconds:.1f}s")
        """

        # Validate variables
        available_vars = self.list_variables()
        invalid_vars = [v for v in variables if v not in available_vars]
        if invalid_vars:
            raise ValueError(
                f"Invalid variables: {invalid_vars}. "
                f"Available variables: {available_vars}"
            )

        result = self.create_download_result()

        try:
            # Step 1: Get ALL available cycles
            self.logger.info("Discovering all available cycles...")
            all_cycles = self._get_all_cycles()
            self.logger.info(f"Found {len(all_cycles)} cycles")

            # Step 2: For each cycle, get ALL passes and check which intersect area
            matching_files = []
            for cycle in all_cycles:
                self.logger.info(f"Checking cycle {cycle}...")
                cycle_files = self._find_files_in_area(
                    cycle=cycle,
                    lon_range=lon_range,
                    lat_range=lat_range,
                )
                matching_files.extend(cycle_files)

            self.logger.info(f"Found {len(matching_files)} files matching area")

            if not matching_files:
                self.logger.warning("No files found matching the specified area")
                return self.finalize_download_result(
                    result, "No files found matching area"
                )

            # Step 3: Download and subset
            downloaded_files = self._download_and_subset_files(
                files=matching_files,
                variables=variables,
                lon_range=lon_range,
                lat_range=lat_range,
                force=force,
                dry_run=dry_run,
                result=result,
            )

            result.downloaded_files = downloaded_files

            return self.finalize_download_result(result)

        except Exception as e:
            result.add_error("download_operation", e)
            return self.finalize_download_result(result)

    def _get_all_cycles(self) -> List[int]:
        """Get all available cycle numbers from FTP."""
        cycles = []
        try:
            with ftplib.FTP(self.ftp_server) as ftp:
                ftp.login(self.username, self.password)
                ftp.cwd(self._ftp_base_path)

                # List all cycle directories
                items = ftp.nlst()
                for item in items:
                    if item.startswith("cycle_"):
                        try:
                            cycle_num = int(item.split("_")[1])
                            cycles.append(cycle_num)
                        except (ValueError, IndexError):
                            continue

                cycles.sort()
                self.logger.debug(f"Found cycles: {cycles}")

        except Exception as e:
            self.logger.error(f"Error getting cycles: {e}")
            raise

        return cycles

    def _find_files_in_area(
        self,
        cycle: int,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """
        Find all files in a cycle that intersect the area.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts with: {'cycle': int, 'pass': int, 'filename': str}
        """
        matching_files = []

        try:
            with ftplib.FTP(self.ftp_server) as ftp:
                ftp.login(self.username, self.password)
                ftp.cwd(self._ftp_base_path)
                ftp.cwd(f"cycle_{cycle:03d}")

                # Get ALL files for this cycle
                cycle_str = f"{cycle:03d}"
                pattern = f"SWOT_{self._level}_LR_SSH_{self._variant}_{cycle_str}_*"
                all_files = ftp.nlst(pattern)

                if not all_files:
                    return matching_files

                # Group by pass number and get latest version for each
                # Note: L3 files don't have versions, so we just take all files
                # For L2, we would need to select latest version
                passes_dict = {}
                for filename in all_files:
                    # Extract pass number: SWOT_L3_LR_SSH_Expert_019_001_...
                    parts = filename.split("_")
                    if len(parts) >= 6:
                        try:
                            pass_num = int(parts[5])
                            # For L3, no version handling needed (only_last=False in notebook)
                            # Just keep one file per pass (take first or any)
                            if pass_num not in passes_dict:
                                passes_dict[pass_num] = filename
                        except (ValueError, IndexError):
                            continue

                # Check each pass file if it intersects area
                for pass_num, filename in passes_dict.items():
                    if self._file_intersects_area(ftp, filename, lon_range, lat_range):
                        matching_files.append(
                            {
                                "cycle": cycle,
                                "pass": pass_num,
                                "filename": filename,
                            }
                        )

        except ftplib.error_perm as e:
            self.logger.warning(f"Error accessing cycle {cycle}: {e}")
        except Exception as e:
            self.logger.warning(f"Error processing cycle {cycle}: {e}")

        return matching_files

    def _file_intersects_area(
        self,
        ftp: ftplib.FTP,
        filename: str,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
    ) -> bool:
        """
        Check if file intersects area by downloading a small sample.

        Parameters
        ----------
        ftp : ftplib.FTP
            FTP connection
        filename : str
            Filename to check
        lon_range : Tuple[float, float]
            Longitude range
        lat_range : Tuple[float, float]
            Latitude range

        Returns
        -------
        bool
            True if file intersects area, False otherwise
        """

        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                tmp_path = tmp.name
                try:
                    ftp.retrbinary(f"RETR {filename}", tmp.write)
                    tmp.close()

                    # Quick check: open and check bounds
                    with xr.open_dataset(tmp_path) as ds:
                        if "longitude" not in ds or "latitude" not in ds:
                            return True  # Can't check, assume yes

                        lon = ds.longitude.values
                        lat = ds.latitude.values

                        # Normalize longitude
                        lon = lon.copy()
                        lon[lon < -180] += 360
                        lon[lon > 180] -= 360

                        lon_min, lon_max = float(lon.min()), float(lon.max())
                        lat_min, lat_max = float(lat.min()), float(lat.max())

                        # Check intersection
                        intersects = (
                            lon_max >= lon_range[0]
                            and lon_min <= lon_range[1]
                            and lat_max >= lat_range[0]
                            and lat_min <= lat_range[1]
                        )

                        return intersects

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        except Exception as e:
            self.logger.debug(f"Error checking {filename}: {e}")
            # If we can't check, assume it might intersect (conservative)
            return True

    def _download_and_subset_files(
        self,
        files: List[Dict[str, Any]],
        variables: List[str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        force: bool,
        dry_run: bool,
        result: DownloadResult,
    ) -> List[str]:
        """Download and subset all matching files."""

        subset_files = []

        for file_info in files:
            cycle = file_info["cycle"]
            filename = file_info["filename"]

            # Full path for downloaded file
            local_path = os.path.join(self.base_path_to_download, filename)

            # Check if already exists
            if not force and os.path.exists(local_path):
                result.add_skipped(local_path, "Already downloaded")
                # Still subset it if subset doesn't exist
                subset_path = os.path.join(
                    self.base_path_to_download, f"subset_{filename}"
                )
                if not os.path.exists(subset_path):
                    subset_path = self._subset_file(
                        local_path,
                        variables,
                        lon_range,
                        lat_range,
                        self.base_path_to_download,
                        result,
                    )
                    if subset_path:
                        subset_files.append(subset_path)
                else:
                    subset_files.append(subset_path)
                continue

            if dry_run:
                subset_path = os.path.join(
                    self.base_path_to_download, f"subset_{filename}"
                )
                result.add_downloaded(subset_path)
                subset_files.append(subset_path)
                continue

            # Download file
            try:
                with ftplib.FTP(self.ftp_server) as ftp:
                    ftp.login(self.username, self.password)
                    ftp.cwd(self._ftp_base_path)
                    ftp.cwd(f"cycle_{cycle:03d}")

                    os.makedirs(self.base_path_to_download, exist_ok=True)
                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {filename}", f.write)

                    result.add_downloaded(local_path)
                    self.logger.info(f"Downloaded: {filename}")

            except Exception as e:
                result.add_error(local_path, e)
                continue

            # Subset file
            subset_path = self._subset_file(
                local_path,
                variables,
                lon_range,
                lat_range,
                self.base_path_to_download,
                result,
            )
            if subset_path:
                subset_files.append(subset_path)

        return subset_files

    def _subset_file(
        self,
        filepath: str,
        variables: List[str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        output_dir: str,
        result: DownloadResult,
    ) -> Optional[str]:
        """
        Subset a single file by area.

        Follows the exact logic from the notebook:
        1. Load dataset and select variables
        2. Create normalized copy for mask calculation
        3. Apply mask to original dataset (not normalized)
        """

        try:
            self.logger.info(f"Subset dataset: {os.path.basename(filepath)}")

            # Open dataset and select variables (as in notebook)
            swot_ds = xr.open_dataset(filepath)
            swot_ds = swot_ds[variables]
            swot_ds.load()

            # Create normalized copy for mask calculation (as in notebook)
            ds = self._normalize_longitude(swot_ds.copy(), -180, 180)

            # Create mask from normalized dataset
            mask = (
                (ds.longitude <= lon_range[1])
                & (ds.longitude >= lon_range[0])
                & (ds.latitude <= lat_range[1])
                & (ds.latitude >= lat_range[0])
            ).compute()

            # Apply mask to ORIGINAL dataset (not normalized) - as in notebook
            swot_ds_area = swot_ds.where(mask, drop=True)

            # Check if empty (as in notebook)
            if swot_ds_area.sizes.get("num_lines", 0) == 0:
                self.logger.warning(
                    f"Dataset {os.path.basename(filepath)} not matching geographical area."
                )
                return None

            # Set compression (as in notebook)
            for var in list(swot_ds_area.keys()):
                swot_ds_area[var].encoding = {"zlib": True, "complevel": 5}

            # Get basename (handle both local paths and URLs like notebook)
            filename = os.path.basename(urlparse(filepath).path)
            subset_filename = f"subset_{filename}"

            self.logger.info(f"Store subset: {subset_filename}")
            subset_path = os.path.join(output_dir, subset_filename)
            swot_ds_area.to_netcdf(subset_path, mode="w")

            return subset_path

        except Exception as e:
            result.add_error(filepath, e)
            return None

    def _normalize_longitude(
        self, ds: xr.Dataset, lon_min: float, lon_max: float
    ) -> xr.Dataset:
        """Normalize longitude values."""

        lon = ds.longitude.values.copy()
        lon[lon < lon_min] += 360
        lon[lon > lon_max] -= 360
        ds = ds.copy()
        ds.longitude.values = lon

        return ds

import gzip
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from ...core.logging import get_file_logger

# Set font sizes for plots
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 12
TEXT_SIZE = 14


class NOAADataReader:
    """
    Class for reading and processing NOAA buoy data.

    This class provides methods to read and process different types of NOAA buoy data,
    including bulk parameters, wave spectra, and directional spectra.

    Attributes
    ----------
    base_path : Path
        Base path where the data is stored
    debug : bool
        Whether to run in debug mode
    """

    def __init__(self, base_path: Union[str, Path], debug: bool = True):
        """
        Initialize the NOAA data reader.

        Parameters
        ----------
        base_path : Union[str, Path]
            Base path where the data is stored
        debug : bool, optional
            Whether to run in debug mode, by default True
        """

        self.base_path = Path(base_path)
        self.debug = debug
        self.logger = get_file_logger(
            "NOAADataReader", level="DEBUG" if debug else "INFO"
        )

    def read_bulk_parameters(self, buoy_id: str, year: int) -> Optional[pd.DataFrame]:
        """
        Read bulk parameters for a specific buoy and year.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        year : int
            The year to read data for

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the bulk parameters, or None if data not found
        """

        file_path = self.base_path / buoy_id / f"buoy_{buoy_id}_bulk_parameters.csv"
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
            return df
        except FileNotFoundError:
            self.logger.error(
                f"No bulk parameters file found for buoy {buoy_id} year {year}"
            )
            return None

    def read_wave_spectra(self, buoy_id: str, year: int) -> Optional[pd.DataFrame]:
        """
        Read wave spectra data for a specific buoy and year.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        year : int
            The year to read data for

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the wave spectra, or None if data not found
        """

        file_path = (
            self.base_path
            / buoy_id
            / "wave_spectra"
            / f"buoy_{buoy_id}_spectra_{year}.csv"
        )
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        except FileNotFoundError:
            self.logger.error(
                f"No wave spectra file found for buoy {buoy_id} year {year}"
            )
            return None

    def read_directional_spectra(
        self, buoy_id: str, year: int
    ) -> Tuple[Optional[pd.DataFrame], ...]:
        """
        Read directional spectra data for a specific buoy and year.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        year : int
            The year to read data for

        Returns
        -------
        Tuple[Optional[pd.DataFrame], ...]
            Tuple containing DataFrames for alpha1, alpha2, r1, r2, and c11,
            or None for each if data not found
        """

        dir_path = self.base_path / buoy_id / "directional_spectra"
        files = {
            "alpha1": f"{buoy_id}d{year}.txt.gz",
            "alpha2": f"{buoy_id}i{year}.txt.gz",
            "r1": f"{buoy_id}j{year}.txt.gz",
            "r2": f"{buoy_id}k{year}.txt.gz",
            "c11": f"{buoy_id}w{year}.txt.gz",
        }

        results = {}
        for name, filename in files.items():
            file_path = dir_path / filename
            try:
                results[name] = self._read_directional_file(file_path)
            except FileNotFoundError:
                self.logger.error(
                    f"No {name} file found for buoy {buoy_id} year {year}"
                )
                results[name] = None

        return (
            results["alpha1"],
            results["alpha2"],
            results["r1"],
            results["r2"],
            results["c11"],
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
                    if not line.startswith("#"):
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

    def plot_bulk_parameters(
        self, buoy_id: str, start_date: str, end_date: str
    ) -> None:
        """
        Plot bulk parameters for a specific buoy and date range.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        """

        # Implementation of plot_bulk_parameters...
        pass

    def plot_wave_spectra(self, buoy_id: str, start_date: str, end_date: str) -> None:
        """
        Plot wave spectra for a specific buoy and date range.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        """

        # Implementation of plot_wave_spectra...
        pass

    def plot_directional_spectra(
        self, buoy_id: str, start_date: str, end_date: str
    ) -> None:
        """
        Plot directional spectra for a specific buoy and date range.

        Parameters
        ----------
        buoy_id : str
            The buoy ID
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        """

        # Implementation of plot_directional_spectra...
        pass

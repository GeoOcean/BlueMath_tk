import os.path as op
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bluemath_tk.downloaders.noaa.noaa_downloader import NOAADownloader


class TestNOAADownloader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = NOAADownloader(
            base_path_to_download=self.temp_dir,
            debug=True,
            check=False,  # Just check paths to download
        )

    def test_download_bulk_parameters(self):
        """Test downloading bulk parameters."""

        # Test without loading DataFrame
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        print(f"\nBulk parameters download result: {result}")

        # Test with loading DataFrame
        df = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
            load_df=True,
        )
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("datetime" in df.columns)
        self.assertTrue(len(df) > 0)
        print(f"\nBulk parameters DataFrame shape: {df.shape}")

    def test_download_wave_spectra(self):
        """Test downloading wave spectra."""

        # Test without loading DataFrame
        result = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        print(f"\nWave spectra download result: {result}")

        # Test with loading DataFrame
        df = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2023],
            load_df=True,
        )
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        self.assertTrue(len(df) > 0)
        print(f"\nWave spectra DataFrame shape: {df.shape}")

    def test_download_directional_spectra(self):
        """Test downloading directional spectra."""

        # Test without loading DataFrame
        result = self.downloader.download_data(
            data_type="directional_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        print(f"\nDirectional spectra download result: {result}")

        # Test with loading DataFrame
        alpha1, alpha2, r1, r2, c11 = self.downloader.download_data(
            data_type="directional_spectra",
            buoy_id="41001",
            years=[2023],
            load_df=True,
        )
        # Check each coefficient DataFrame
        for name, df in [
            ("alpha1", alpha1),
            ("alpha2", alpha2),
            ("r1", r1),
            ("r2", r2),
            ("c11", c11),
        ]:
            if df is not None:
                self.assertIsInstance(df, pd.DataFrame)
                self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
                self.assertTrue(len(df) > 0)
                print(f"\n{name} DataFrame shape: {df.shape}")

    def test_multiple_years_loading(self):
        """Test loading multiple years of data."""

        # Test bulk parameters with multiple years
        df = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2022, 2023],
            load_df=True,
        )
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("datetime" in df.columns)
        self.assertTrue(len(df) > 0)

        # Check that data spans multiple years
        years = df["datetime"].dt.year.unique()
        self.assertTrue(len(years) > 1)
        print(f"\nBulk parameters multiple years: {sorted(years)}")

        # Test wave spectra with multiple years
        df = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2022, 2023],
            load_df=True,
        )
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        self.assertTrue(len(df) > 0)

        # Check that data spans multiple years
        years = df.index.year.unique()
        self.assertTrue(len(years) > 1)
        print(f"\nWave spectra multiple years: {sorted(years)}")

    def test_list_data_types(self):
        """Test listing available data types."""

        data_types = self.downloader.list_data_types()
        self.assertIsInstance(data_types, list)
        self.assertTrue(len(data_types) > 0)
        print(f"\nAvailable data types: {data_types}")

    def test_list_datasets(self):
        """Test listing available datasets."""

        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        print(f"\nAvailable datasets: {datasets}")

    def test_show_markdown_table(self):
        """Test showing markdown table."""

        self.downloader.show_markdown_table()

    def test_file_paths(self):
        """Test that downloaded files exist in the correct locations."""

        # Download data
        self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
        )

        # Check bulk parameters file
        bulk_file = op.join(
            self.temp_dir,
            "buoy_data",
            "41001",
            "buoy_41001_bulk_parameters.csv",
        )
        self.assertTrue(op.exists(bulk_file))
        print(f"\nBulk parameters file exists: {bulk_file}")

        # Download and check wave spectra
        self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2023],
        )
        wave_file = op.join(
            self.temp_dir,
            "buoy_data",
            "41001",
            "wave_spectra",
            "buoy_41001_spectra_2023.csv",
        )
        self.assertTrue(op.exists(wave_file))
        print(f"\nWave spectra file exists: {wave_file}")

        # Download and check directional spectra
        self.downloader.download_data(
            data_type="directional_spectra",
            buoy_id="41001",
            years=[2023],
        )
        dir_path = op.join(
            self.temp_dir,
            "buoy_data",
            "41001",
            "directional_spectra",
        )
        self.assertTrue(op.exists(dir_path))
        # Check for at least one coefficient file
        coeff_files = list(Path(dir_path).glob("41001*2023.txt.gz"))
        self.assertTrue(len(coeff_files) > 0)
        print(f"\nDirectional spectra files exist: {coeff_files}")


if __name__ == "__main__":
    unittest.main()

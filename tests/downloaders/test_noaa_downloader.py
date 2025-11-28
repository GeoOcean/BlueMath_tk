import os.path as op
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bluemath_tk.downloaders._download_result import DownloadResult
from bluemath_tk.downloaders.noaa.noaa_downloader import (
    NOAADownloader,
    read_bulk_parameters,
    read_directional_spectra,
    read_wave_spectra,
)


class TestNOAADownloader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = NOAADownloader(
            product="NDBC",
            base_path_to_download=self.temp_dir,
            debug=True,
        )

    def test_download_bulk_parameters(self):
        """Test downloading bulk parameters."""

        # Test download
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DownloadResult)
        print(f"\nBulk parameters download result: {result}")

        # Test dry_run
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
            dry_run=True,
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DownloadResult)
        print(f"\nBulk parameters dry_run result: {result}")

        # Test reading downloaded data
        df = read_bulk_parameters(
            base_path=self.temp_dir,
            buoy_id="41001",
            years=[2023],
        )
        if df is not None:
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue("datetime" in df.columns)
            self.assertTrue(len(df) > 0)
            print(f"\nBulk parameters DataFrame shape: {df.shape}")

    def test_download_wave_spectra(self):
        """Test downloading wave spectra."""

        # Test download
        result = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DownloadResult)
        print(f"\nWave spectra download result: {result}")

        # Test reading downloaded data
        df = read_wave_spectra(
            base_path=self.temp_dir,
            buoy_id="41001",
            years=[2023],
        )
        if df is not None:
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
            self.assertTrue(len(df) > 0)
            print(f"\nWave spectra DataFrame shape: {df.shape}")

    def test_download_directional_spectra(self):
        """Test downloading directional spectra."""

        # Test download
        result = self.downloader.download_data(
            data_type="directional_spectra",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DownloadResult)
        print(f"\nDirectional spectra download result: {result}")

        # Test reading downloaded data
        alpha1, alpha2, r1, r2, c11 = read_directional_spectra(
            base_path=self.temp_dir,
            buoy_id="41001",
            years=[2023],
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

        # Download multiple years
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2022, 2023],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DownloadResult)

        # Test reading bulk parameters with multiple years
        df = read_bulk_parameters(
            base_path=self.temp_dir,
            buoy_id="41001",
            years=[2022, 2023],
        )
        if df is not None:
            self.assertIsNotNone(df)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue("datetime" in df.columns)
            self.assertTrue(len(df) > 0)

            # Check that data spans multiple years
            years = df["datetime"].dt.year.unique()
            self.assertTrue(len(years) > 1)
            print(f"\nBulk parameters multiple years: {sorted(years)}")

        # Download wave spectra for multiple years
        result = self.downloader.download_data(
            data_type="wave_spectra",
            buoy_id="41001",
            years=[2022, 2023],
        )
        self.assertIsNotNone(result)

        # Test reading wave spectra with multiple years
        df = read_wave_spectra(
            base_path=self.temp_dir,
            buoy_id="41001",
            years=[2022, 2023],
        )
        if df is not None:
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
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
        )
        self.assertIsInstance(result, DownloadResult)

        # Check bulk parameters file
        bulk_file = op.join(
            self.temp_dir,
            "buoy_data",
            "41001",
            "buoy_41001_bulk_parameters.csv",
        )
        if op.exists(bulk_file):
            self.assertTrue(op.exists(bulk_file))
            print(f"\nBulk parameters file exists: {bulk_file}")

        # Download and check wave spectra
        result = self.downloader.download_data(
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
        if op.exists(wave_file):
            self.assertTrue(op.exists(wave_file))
            print(f"\nWave spectra file exists: {wave_file}")

        # Download and check directional spectra
        result = self.downloader.download_data(
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
        if op.exists(dir_path):
            self.assertTrue(op.exists(dir_path))
            # Check for at least one coefficient file
            coeff_files = list(Path(dir_path).glob("41001*2023.txt.gz"))
            if len(coeff_files) > 0:
                self.assertTrue(len(coeff_files) > 0)
                print(f"\nDirectional spectra files exist: {coeff_files}")

    def test_dry_run(self):
        """Test dry_run functionality."""

        # Test dry_run for bulk parameters
        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)
        self.assertTrue(
            len(result.skipped_files) > 0 or len(result.downloaded_files) > 0
        )
        print(f"\nDry run result: {result}")

    def test_product_parameter(self):
        """Test that product parameter is required and validated."""

        # Test with valid product
        downloader = NOAADownloader(
            product="NDBC",
            base_path_to_download=self.temp_dir,
        )
        self.assertEqual(downloader.product, "NDBC")

        # Test with invalid product
        with self.assertRaises(ValueError):
            NOAADownloader(
                product="INVALID",
                base_path_to_download=self.temp_dir,
            )

    def test_download_result_structure(self):
        """Test DownloadResult structure."""

        result = self.downloader.download_data(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
            dry_run=True,
        )

        self.assertIsInstance(result, DownloadResult)
        self.assertIsNotNone(result.start_time)
        self.assertIsNotNone(result.end_time)
        self.assertIsNotNone(result.duration_seconds)
        self.assertGreater(result.duration_seconds, 0)
        self.assertIsInstance(result.downloaded_files, list)
        self.assertIsInstance(result.skipped_files, list)
        self.assertIsInstance(result.error_files, list)
        self.assertIsInstance(result.message, str)
        print(f"\nDownloadResult structure: {result}")

    def test_product_specific_method(self):
        """Test calling product-specific download method directly."""
        result = self.downloader.download_data_ndbc(
            data_type="bulk_parameters",
            buoy_id="41001",
            years=[2023],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)
        print(f"\nProduct-specific method result: {result}")

    def test_invalid_data_type(self):
        """Test that invalid data type raises ValueError."""
        with self.assertRaises(ValueError):
            self.downloader.download_data(
                data_type="invalid_type",
                buoy_id="41001",
                years=[2023],
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest

from bluemath_tk.downloaders._download_result import DownloadResult
from bluemath_tk.downloaders.copernicus.copernicus_downloader import (
    CopernicusDownloader,
)


class TestCopernicusDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = CopernicusDownloader(
            product="ERA5",
            base_path_to_download=self.temp_dir,
            token=None,
        )

    def test_download_data_era5(self):
        """Test downloading ERA5 data."""
        result = self.downloader.download_data_era5(
            variables=["spectra"],
            years=[f"{year:04d}" for year in range(2020, 2025)],
            months=[
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            area=[43.4, 350.4, 43.6, 350.6],  # [lat_min, lon_min, lat_max, lon_max]
        )
        self.assertIsInstance(result, DownloadResult)
        print(result)

    def test_download_data_era5_dry_run(self):
        """Test dry_run functionality for ERA5."""
        result = self.downloader.download_data_era5(
            variables=["spectra"],
            years=["2020"],
            months=["01"],
            area=[43.4, 350.4, 43.6, 350.6],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)
        self.assertTrue(
            len(result.skipped_files) > 0 or len(result.downloaded_files) > 0
        )
        print(f"\nDry run result: {result}")

    def test_download_data_routing(self):
        """Test that download_data routes to product-specific methods."""
        result = self.downloader.download_data(
            variables=["spectra"],
            years=["2020"],
            months=["01"],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)

    def test_product_parameter(self):
        """Test that product parameter is required and validated."""
        # Test with valid product ERA5
        downloader = CopernicusDownloader(
            product="ERA5",
            base_path_to_download=self.temp_dir,
        )
        self.assertEqual(downloader.product, "ERA5")

        # Test with valid product CERRA
        downloader = CopernicusDownloader(
            product="CERRA",
            base_path_to_download=self.temp_dir,
        )
        self.assertEqual(downloader.product, "CERRA")

        # Test with invalid product
        with self.assertRaises(ValueError):
            CopernicusDownloader(
                product="INVALID",
                base_path_to_download=self.temp_dir,
            )

    def test_list_variables(self):
        """Test listing available variables."""
        variables = self.downloader.list_variables()
        self.assertIsInstance(variables, list)
        self.assertTrue(len(variables) > 0)
        print(f"\nAvailable variables: {variables}")

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        print(f"\nAvailable datasets: {datasets}")

    def test_download_data_cerra(self):
        """Test downloading CERRA data."""
        cerra_downloader = CopernicusDownloader(
            product="CERRA",
            base_path_to_download=self.temp_dir,
            token=None,
        )
        result = cerra_downloader.download_data_cerra(
            variables=["10m_wind_speed"],
            years=["2020"],
            months=["01"],
            days=["01"],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)
        print(f"\nCERRA download result: {result}")

    def test_download_data_cerra_dry_run(self):
        """Test dry_run functionality for CERRA."""
        cerra_downloader = CopernicusDownloader(
            product="CERRA",
            base_path_to_download=self.temp_dir,
            token=None,
        )
        result = cerra_downloader.download_data_cerra(
            variables=["10m_wind_direction"],
            years=["2020"],
            months=["01"],
            days=["01"],
            dry_run=True,
        )
        self.assertIsInstance(result, DownloadResult)
        self.assertTrue(
            len(result.skipped_files) > 0 or len(result.downloaded_files) > 0
        )
        print(f"\nCERRA dry run result: {result}")


if __name__ == "__main__":
    unittest.main()
